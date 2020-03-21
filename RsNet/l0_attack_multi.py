## l0_attack.py -- attack a network optimizing for l_0 distance
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
from RsNet.tf_config import CHANNELS_LAST, CHANNELS_FIRST

BINARY_SEARCH_STEPS = 20  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 1000  # number of iterations to perform gradient descent
ABORT_EARLY = True  # abort gradient descent upon first valid solution
LEARNING_RATE = 1e-2  # larger values converge faster to less accurate results
INITIAL_CONST = 1e-3  # the first value of c to start at
LARGEST_CONST = 2e6  # the largest value of c to go up to before giving up
REDUCE_CONST = False  # try to lower c each iteration; faster to set to false
TARGETED = True  # should we target one specific class? or just be wrong?
CONFIDENCE = 0  # how strong the adversarial example should be
CONST_FACTOR = 2.0  # f>1, rate at which we increase constant, smaller better


class CarliniL0:
    def __init__(self, sess, models, detector_dict, thrs,
                 batch_size=1, confidence=CONFIDENCE,
                 targeted=TARGETED, learning_rate=LEARNING_RATE,
                 binary_search_steps=BINARY_SEARCH_STEPS,
                 max_iterations=MAX_ITERATIONS, abort_early=ABORT_EARLY,
                 initial_const=INITIAL_CONST, largest_const=LARGEST_CONST,
                 reduce_const=REDUCE_CONST, const_factor=CONST_FACTOR,
                 data_format=CHANNELS_LAST,
                 independent_channels=False, boxmin=-0.5, boxmax=0.5):
        """
        The L_0 optimized attack.

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
        const_factor: The rate at which we should increase the constant, when the
          previous constant failed. Should be greater than one, smaller is better.
        independent_channels: set to false optimizes for number of pixels changed,
          set to true (not recommended) returns number of channels changed.
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
        self.REDUCE_CONST = reduce_const
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.const_factor = const_factor
        self.independent_channels = independent_channels
        self.batch_size = batch_size
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
        self.canchange = tf.Variable(np.zeros(shape), dtype=tf.float32, name='canchange')
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32, name='timg')
        self.simg = tf.Variable(np.zeros(shape), dtype=tf.float32, name='simg')
        self.original = tf.Variable(np.zeros(shape, dtype=np.float32), name='oimg')
        self.tlab = tf.Variable(np.zeros((batch_size, num_labels)), dtype=tf.float32, name='tlab')
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32, name='const')
        self.const2 = tf.Variable(np.zeros(batch_size), dtype=tf.float32, name='const2')

        # and here's what we use to assign them
        self.assign_canchange = tf.placeholder(tf.float32, shape)
        self.assign_timg = tf.placeholder(tf.float32, shape)
        self.assign_simg = tf.placeholder(tf.float32, shape)
        self.assign_original = tf.placeholder(np.float32, shape)
        self.assign_tlab = tf.placeholder(tf.float32, (batch_size, num_labels))
        self.assign_const = tf.placeholder(tf.float32, [batch_size])
        self.assign_const2 = tf.placeholder(tf.float32, [batch_size])

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.canchange.assign(self.assign_canchange))
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.simg.assign(self.assign_simg))
        self.setup.append(self.original.assign(self.assign_original))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))
        self.setup.append(self.const2.assign(self.assign_const2))

        self.newimg = (tf.tanh(modifier + self.simg) * self.boxmul + self.boxplus) * self.canchange \
                      + (1 - self.canchange) * self.original

        self.output = [model.predict(self.newimg) for model in models]
        print(self.output[0].shape)
        self.output = tf.stack(self.output, axis=1)
        print(self.output.shape)

        tlab = self.tlab[:, tf.newaxis, :]
        # compute the probability of the label class versus the maximum other
        real = tf.reduce_sum(tlab * self.output, 2)
        other = tf.reduce_max((1 - tlab) * self.output - (tlab * 10000), 2)

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other - real + self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, real - other + self.CONFIDENCE)

        loss1 = tf.reduce_sum(loss1, axis=1)

        loss2 = tf.reduce_sum(tf.square(self.newimg - (tf.tanh(self.timg) * self.boxmul + self.boxplus)), [1, 2, 3])

        # sum up the losses
        detected = [detector.tf_mark(self.newimg, data_format=data_format) - thrs[name]
                    for name, detector in detector_dict.items()]
        detected = tf.maximum(0.0, tf.stack(detected, 0))

        self.detected = tf.reduce_sum(detected, axis=0)

        self.loss3 = tf.reduce_sum(self.const2 * self.detected)  # TODO one const per detector
        self.loss2_ind = loss2
        self.loss1_ind = self.const * loss1
        self.loss2 = tf.reduce_sum(self.loss2_ind)
        self.loss1 = tf.reduce_sum(self.loss1_ind)
        self.loss_ind = self.loss1_ind + self.loss2_ind
        self.loss = self.loss1 + self.loss2 + self.loss3

        self.outgrad = tf.gradients(self.loss, [modifier])[0]

        # setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, var_list=[modifier])

        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        self.init = tf.variables_initializer(var_list=[modifier] + new_vars)

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

        # the pixels we can change
        valid = np.ones(imgs.shape)
        cur_valid_count = prev_valid_count = np.sum(valid)

        # the best li, score, and image attack
        o_bestl0 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = [np.zeros(imgs[0].shape)] * batch_size

        out_step = 0
        # loop does not end until const exceed the upper bound or no pixels changed
        while np.sum(const > self.LARGEST_CONST) < batch_size or cur_valid_count < prev_valid_count:
            if out_step >= self.BINARY_SEARCH_STEPS:
                break

            out_step += 1

            # initialize the variables
            self.sess.run(self.init)

            batch = timgs
            batchlab = labs
            sbatch = simgs

            bestl0 = [1e10] * batch_size
            bestscore = [-1] * batch_size

            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_timg: batch,
                                       self.assign_simg: sbatch,
                                       self.assign_original: imgs,
                                       self.assign_tlab: batchlab,
                                       self.assign_const: const,
                                       self.assign_const2: CONST2 * 10000,
                                       self.assign_canchange: valid})

            prev = 1e10

            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack
                _, l, l_ind, l0s, scores, nimg, det, gradientnorm = \
                    self.sess.run([self.train, self.loss, self.loss_ind,
                                   self.loss2_ind, self.output, self.newimg,
                                   self.detected, self.outgrad])

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
                for e, (l0, sc, ii, detected) in enumerate(zip(l0s, scores, nimg, det)):
                    #    continue
                    if l0 < bestl0[e] and all([compare(scc, np.argmax(batchlab[e])) for scc in sc]) and detected == 0:
                        bestl0[e] = l0
                        bestscore[e] = np.argmax(sc) % self.num_labels
                        o_bestattack[e] = ii

                # abort early if find solution for all instances
                if self.ABORT_EARLY and np.all(np.array(bestscore) != -1):
                    print("find all solution")
                    break

            print("old const")
            print(const)

            if self.independent_channels:
                equal_count = self.model.image_size ** 2 - np.sum(np.abs(nimg - imgs) < .0001, (1, 2, 3))
                print("Forced equal:\n", np.sum(1 - valid, (1, 2, 3)))
                # we are allowed to change each channel independently
                valid = valid.reshape((batch_size, -1))
                totalchange = np.abs(nimg - imgs) * np.abs(gradientnorm)

            else:
                if self.data_format == CHANNELS_FIRST:
                    valid = valid.transpose([0, 2, 3, 1])
                    axis = 1
                else:
                    axis = 3
                equal_count = self.model.image_size ** 2 - np.sum(np.all(np.abs(nimg - imgs) < .0001, axis=axis), (1, 2))
                print("Forced equal:\n", np.sum(np.all(1 - valid, axis=3), (1, 2)))
                # we care only about which pixels change, not channels independently
                # compute total change as sum of change for each channel
                valid = valid.reshape((batch_size, self.model.image_size ** 2, self.model.num_channels))
                totalchange = np.abs(np.sum(nimg - imgs, axis=axis)) * np.sum(np.abs(gradientnorm), axis=axis)

            print("Equal count:\n", equal_count)
            totalchange = totalchange.reshape((batch_size, -1))

            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and bestscore[e] != -1:

                    # set some of the pixels to 0 depending on their total change
                    did = 0
                    cur_totalchange = totalchange[e]
                    cur_valid = valid[e]
                    cur_equal_count = equal_count[e]
                    for i in np.argsort(cur_totalchange):

                        if np.all(cur_valid[i]):
                            did += 1
                            cur_valid[i] = 0

                            if cur_totalchange[i] > .01:
                                # if this pixel changed a lot, skip
                                break
                            if did >= .25 * cur_equal_count:
                                # if we changed too many pixels, skip
                                break
                    # use warm start
                    simgs[e] = np.arctanh((nimg[e] - self.boxplus) / self.boxmul * 0.999999)

                else:
                    if const[e] < self.LARGEST_CONST:
                        const[e] *= self.const_factor

            print("updated const:\n", const)
            valid = np.reshape(valid, (batch_size, self.model.image_size, self.model.image_size, -1))
            if self.independent_channels:
                print("Now forced equal:\n", np.sum(1 - valid, (1, 2, 3)))
            else:
                print("Forced equal:\n", np.sum(np.all(1 - valid, axis=3), (1, 2)))
                if self.data_format == CHANNELS_FIRST:
                    valid = valid.transpose([0, 3, 1, 2])

            prev_valid_count = cur_valid_count
            cur_valid_count = np.sum(valid)

        # return the best solution found
        o_bestl0 = np.array(o_bestl0)
        return o_bestattack
