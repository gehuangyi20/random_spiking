## li_attack.py -- attack a network optimizing for l_infinity distance
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
from RsNet.tf_config import CHANNELS_LAST, CHANNELS_FIRST

BINARY_SEARCH_STEPS = 15  # number of times to adjust the constant with binary search
DECREASE_FACTOR = 0.9   # 0<f<1, rate at which we shrink tau; larger is more accurate
MAX_ITERATIONS = 1000   # number of iterations to perform gradient descent
ABORT_EARLY = True      # abort gradient descent upon first valid solution
INITIAL_CONST = 1e-5    # the first value of c to start at
LEARNING_RATE = 5e-3    # larger values converge faster to less accurate results
LARGEST_CONST = 2e+1    # the largest value of c to go up to before giving up
REDUCE_CONST = False    # try to lower c each iteration; faster to set to false
TARGETED = True         # should we target one specific class? or just be wrong?
CONFIDENCE = 0          # how strong the adversarial example should be
CONST_FACTOR = 2.0      # f>1, rate at which we increase constant, smaller better


class CarliniLi:
    def __init__(self, sess, models, detector_dict, thrs,
                 batch_size=1, confidence=CONFIDENCE,
                 targeted=TARGETED, learning_rate=LEARNING_RATE,
                 binary_search_steps=BINARY_SEARCH_STEPS,
                 max_iterations=MAX_ITERATIONS, abort_early=ABORT_EARLY,
                 initial_const=INITIAL_CONST, largest_const=LARGEST_CONST,
                 reduce_const=REDUCE_CONST, decrease_factor=DECREASE_FACTOR,
                 data_format=CHANNELS_LAST,
                 const_factor=CONST_FACTOR, warm_start=True, boxmin=-0.5, boxmax=0.5):
        """
        The L_infinity optimized attack. 

        Returns adversarial examples for the supplied model.

        batch_size: Number of attacks to run simultaneously.
        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. Should be set to a very small
          value (but positive).
        largest_const: The largest constant to use until we report failure. Should
          be set to a very large value.
        reduce_const: If true, after each successful attack, make const smaller.
        decrease_factor: Rate at which we should decrease tau, less than one.
          Larger produces better quality results.
        const_factor: The rate at which we should increase the constant, when the
          previous constant failed. Should be greater than one, smaller is better.
        warm_start: whether generating the adv img based on prev iteration
        boxmin: Minimum pixel value (default -0.5).
        boxmax: Maximum pixel value (default 0.5).
        """

        image_size, num_channels, num_labels = models[0].image_size, models[0].num_channels, models[0].num_labels

        self.model = models[0]
        self.sess = sess

        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.INITIAL_CONST = initial_const
        self.LARGEST_CONST = largest_const
        self.DECREASE_FACTOR = decrease_factor
        self.REDUCE_CONST = reduce_const
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.const_factor = const_factor
        self.batch_size = batch_size
        self.warm_start = warm_start
        self.boxmul = (boxmax - boxmin) / 2.
        self.boxplus = (boxmin + boxmax) / 2.
        self.num_labels = models[0].num_labels
        self.data_format = data_format

        if self.data_format == CHANNELS_LAST:
            shape = (batch_size, image_size, image_size, num_channels)
        else:
            shape = (batch_size, num_channels, image_size, image_size)

        # the variable to optimize over
        modifier = tf.Variable(tf.zeros(shape, dtype=tf.float32), name='modifier')

        # these are variables to be more efficient in sending data to tf
        self.tau = tf.Variable(np.zeros(batch_size), dtype=tf.float32, name='tau')
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32, name='timg')
        self.simg = tf.Variable(np.zeros(shape), dtype=tf.float32, name='simg')
        self.tlab = tf.Variable(np.zeros((batch_size, num_labels)), dtype=tf.float32, name='tlab')
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32, name='const')
        self.const2 = tf.Variable(np.zeros(batch_size), dtype=tf.float32, name='const2')

        # and here's what we use to assign them
        self.assign_tau = tf.placeholder(tf.float32, [batch_size])
        self.assign_timg = tf.placeholder(tf.float32, shape)
        self.assign_simg = tf.placeholder(tf.float32, shape)
        self.assign_tlab = tf.placeholder(tf.float32, (batch_size, num_labels))
        self.assign_const = tf.placeholder(tf.float32, [batch_size])
        self.assign_const2 = tf.placeholder(tf.float32, [batch_size])

        self.newimg = tf.tanh(modifier + self.simg) * self.boxmul + self.boxplus

        self.output = [model.predict(self.newimg) for model in models]
        print(self.output[0].shape)
        self.output = tf.stack(self.output, axis=1)
        print(self.output.shape)

        tlab = self.tlab[:, tf.newaxis, :]
        # compute the probability of the label class versus the maximum other
        real = tf.reduce_sum(tlab * self.output, 2)
        other = tf.reduce_max((1 - tlab) * self.output - (tlab * 10000), 2)

        # distance to the input data
        tau = self.tau[:, tf.newaxis, tf.newaxis, tf.newaxis]
        lidist = tf.maximum(
            0.0, tf.abs(self.newimg - (tf.tanh(self.timg) * self.boxmul + self.boxplus)) - tau)
        self.lidist = tf.reduce_sum(lidist, [1, 2, 3])

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other - real + self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, real - other + self.CONFIDENCE)

        # print('loss1 before sum', loss1.get_shape())
        loss1 = tf.reduce_sum(loss1, axis=1)
        # print('loss1', loss1.get_shape())

        # sum up the losses
        detected = [detector.tf_mark(self.newimg, data_format=data_format) - thrs[name]
                    for name, detector in detector_dict.items()]
        # print('detected[0]', detected[0].get_shape(), 'len',len(detected))
        detected = tf.maximum(0.0, tf.stack(detected, 0))
        # print('new detected', detected.get_shape())

        self.detected = tf.reduce_sum(detected, axis=0)

        self.loss3 = tf.reduce_sum(self.const2 * self.detected)  # TODO one const per detector
        self.loss2_ind = self.lidist
        self.loss1_ind = self.const * loss1
        self.loss2 = tf.reduce_sum(self.loss2_ind)
        self.loss1 = tf.reduce_sum(self.loss1_ind)
        self.loss_ind = self.loss1_ind + self.loss2_ind
        self.loss = self.loss1 + self.loss2 + self.loss3

        # setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, var_list=[modifier])

        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        self.init = tf.variables_initializer(var_list=[modifier] + new_vars)

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.tau.assign(self.assign_tau))
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.simg.assign(self.assign_simg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))
        self.setup.append(self.const2.assign(self.assign_const2))

    def attack(self, imgs, targets, input_data_format=CHANNELS_LAST, output_data_format=CHANNELS_LAST):
        """
        Perform the L_0 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        assert input_data_format in (CHANNELS_FIRST, CHANNELS_LAST)
        assert output_data_format in (CHANNELS_FIRST, CHANNELS_LAST)
        r = []
        count = len(imgs)
        print('go up to', len(imgs))
        # tranpose the input data format to fit the data format of the attack model
        if input_data_format != self.data_format:
            if input_data_format == CHANNELS_LAST:
                # input is channels_last, transpose to channels_first
                imgs = np.transpose(imgs, [0, 3, 1, 2])
            else:
                # input is channels_first, transpose to channels_last
                imgs = np.transpose(imgs, [0, 2, 3, 1])
        for i in range(0, count, self.batch_size):
            print('tick', i)
            attack_imgs = imgs[i:i + self.batch_size]
            attack_targets = targets[i:i + self.batch_size]
            attack_len = len(attack_targets)

            if attack_len < self.batch_size:
                img_shape = np.asarray(attack_imgs.shape)
                img_shape[0] = self.batch_size - attack_len
                target_shape = np.asarray(attack_targets.shape)
                target_shape[0] = self.batch_size - attack_len
                attack_imgs = np.append(attack_imgs, np.zeros(img_shape), axis=0)
                attack_targets = np.append(attack_targets, np.zeros(target_shape), axis=0)
            r.extend(self.attack_batch(attack_imgs, attack_targets))

        output = np.array(r)[0: count]
        # tranpose the output data format of the attack model to fit the output data format
        if output_data_format != self.data_format:
            if output_data_format == CHANNELS_LAST:
                # attack model output is channels_first, transpose to channels_last
                output = np.transpose(output, [0, 2, 3, 1])
            else:
                # attack model output is channels_last, transpose to channels_first
                output = np.transpose(output, [0, 3, 1, 2])
        return output

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a single image and label
        """

        def compare(x, y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                if self.TARGETED:
                    x[y] -= self.CONFIDENCE
                else:
                    x[y] += self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        batch_size = self.batch_size

        # convert to tanh-space
        timgs = np.arctanh((imgs - self.boxplus) / self.boxmul * 0.999999)
        simgs = np.copy(timgs)

        # set the lower and upper bounds accordingly
        # lower_bound = np.zeros(batch_size)
        const = np.ones(batch_size) * self.INITIAL_CONST
        CONST2 = np.ones(batch_size) * self.INITIAL_CONST
        # upper_bound = np.ones(batch_size) * self.LARGEST_CONST #1e10

        # init tau
        tau = np.ones(batch_size) * 1.0

        # the best li, score, and image attack
        o_bestli = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = [np.zeros(imgs[0].shape)] * batch_size

        out_step = 0
        while np.sum((tau <= 1/256) | (const > self.LARGEST_CONST)) < batch_size:
            if out_step >= self.BINARY_SEARCH_STEPS:
                break

            out_step += 1
            self.sess.run(self.init)

            batch = timgs
            batchlab = labs
            sbatch = simgs

            bestli = [1e10] * batch_size
            bestscore = [-1] * batch_size

            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_timg: batch,
                                       self.assign_simg: sbatch,
                                       self.assign_tlab: batchlab,
                                       self.assign_const: const,
                                       self.assign_const2: CONST2 * 10000,
                                       self.assign_tau: tau})
            prev = 1e10

            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack
                _, l, l_ind, lis, scores, det, nimg = self.sess.run([
                    self.train, self.loss, self.loss_ind,
                    self.lidist, self.output,
                    self.detected,
                    self.newimg])

                # print out the losses every 10%
                if iteration % (self.MAX_ITERATIONS // 10) == 0:
                    print(iteration, self.sess.run((self.loss, self.loss1, self.loss2, self.loss3)))

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and iteration % (self.MAX_ITERATIONS // 10) == 0:
                    if l > prev * .9999 or l == 0:
                        print("gradient descent gets stuck")
                        break
                    prev = l

                # print('detection rate',det)
                # adjust the best result found so far
                if type(det) == np.float32:
                    det = [0] * batch_size
                for e, (li, sc, ii, detected) in enumerate(zip(lis, scores, nimg, det)):
                    if self.warm_start and l_ind[e] > 0.0001*const[e]:
                        continue
                    if li < bestli[e] and all([compare(scc, np.argmax(batchlab[e])) for scc in sc]) and detected == 0:
                        bestli[e] = li
                        bestscore[e] = np.argmax(sc) % self.num_labels
                        o_bestattack[e] = ii
                    # if li < o_bestli[e] and all([compare(scc, np.argmax(batchlab[e])) for scc in sc]):
                    #     o_bestli[e] = li
                    #     o_bestscore[e] = np.argmax(sc) % self.num_labels
                    #     o_bestattack[e] = ii

                # abort early if find solution for all instances
                if self.ABORT_EARLY and np.all(np.array(bestscore) != -1):
                    break


            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and bestscore[e] != -1:
                    actualtau = np.max(np.abs(nimg[e] - imgs[e]))
                    # use warm start
                    if self.warm_start:
                        simgs[e] = np.arctanh((nimg[e] - self.boxplus) / self.boxmul * 0.999999)
                    if actualtau < tau[e]:
                        tau[e] = actualtau
                    if tau[e] > 1/256:
                        tau[e] *= self.DECREASE_FACTOR
                else:
                    if const[e] < self.LARGEST_CONST:
                        const[e] *= self.const_factor

        # return the best solution found
        o_bestli = np.array(o_bestli)
        return o_bestattack
