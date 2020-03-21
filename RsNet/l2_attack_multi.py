## l2_attack.py -- attack a network optimizing for l_2 distance
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
from RsNet.tf_config import CHANNELS_LAST, CHANNELS_FIRST

BINARY_SEARCH_STEPS = 9  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000  # number of iterations to perform gradient descent
ABORT_EARLY = True  # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2  # larger values converge faster to less accurate results
TARGETED = True  # should we target one specific class? or just be wrong?
CONFIDENCE = 0  # how strong the adversarial example should be
INITIAL_CONST = 1e-3  # the initial constant c to pick as a first guess
CONST_FACTOR = 10.0  # f>1, rate at which we increase constant, smaller better


class CarliniL2:
    def __init__(self, sess, models, detector_dict, detector_gpu_idx, thrs,
                 batch_size=1, confidence=CONFIDENCE,
                 targeted=TARGETED, learning_rate=LEARNING_RATE,
                 binary_search_steps=BINARY_SEARCH_STEPS, max_iterations=MAX_ITERATIONS,
                 abort_early=ABORT_EARLY,
                 initial_const=INITIAL_CONST, const_factor=CONST_FACTOR,
                 data_format=CHANNELS_LAST,
                 boxmin=-0.5, boxmax=0.5, temp=1, gpu_count=1, eot_count=1, eot_det_count=1):
        """
        The L_2 optimized attack.

        This attack is the most efficient and should be used as the primary
        attack to evaluate potential defenses.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of attacks to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence.
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        boxmin: Minimum pixel value (default -0.5).
        boxmax: Maximum pixel value (default 0.5).
        """

        image_size, num_channels, num_labels = models[0].image_size, models[0].num_channels, models[0].num_labels
        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.const_factor = const_factor
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.data_format = data_format

        self.repeat = binary_search_steps >= 10

        if self.data_format == CHANNELS_LAST:
            shape = (batch_size, image_size, image_size, num_channels)
        else:
            shape = (batch_size, num_channels, image_size, image_size)

        # the variable we're going to optimize over
        modifier = tf.Variable(np.zeros(shape, dtype=np.float32), name='modifier')

        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32, name='timg')
        self.tlab = tf.Variable(np.zeros((batch_size, num_labels)), dtype=tf.float32, name='tlab')
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32, name='const')
        self.const2 = tf.Variable(np.zeros(batch_size), dtype=tf.float32, name='const2')

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf.float32, shape)
        self.assign_tlab = tf.placeholder(tf.float32, (batch_size, num_labels))
        self.assign_const = tf.placeholder(tf.float32, [batch_size])
        self.assign_const2 = tf.placeholder(tf.float32, [batch_size])

        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        self.boxmul = (boxmax - boxmin) / 2.
        self.boxplus = (boxmin + boxmax) / 2.
        self.newimg = tf.tanh(modifier + self.timg) * self.boxmul + self.boxplus


        # prediction BEFORE-SOFTMAX of the model
        self.output = []
        loss1 = []
        for i in range(gpu_count):
            loss1.append([])
        gpu_i = 0
        for model in models:
            cur_gpu_idx = gpu_i % gpu_count
            with tf.device('/gpu:' + str(cur_gpu_idx)):
                eot_loss1 = []
                eot_output = []
                for eot_i in range(eot_count):
                    _x = self.newimg
                    tmp_output = model.predict(_x)

                    if temp != 1:
                        tmp_output /= temp
                    eot_output.append(tmp_output)

                    tlab = self.tlab
                    # compute the probability of the label class versus the maximum other
                    real = tf.reduce_sum(tlab * tmp_output, 1)
                    other = tf.reduce_max((1 - tlab) * tmp_output - (tlab * 10000), 1)


                    if self.TARGETED:
                        # if targetted, optimize for making the other class most likely
                        tmp_loss1 = tf.maximum(0.0, other - real + self.CONFIDENCE)
                    else:
                        # if untargeted, optimize for making this class least likely.
                        tmp_loss1 = tf.maximum(0.0, real - other + self.CONFIDENCE)

                    eot_loss1.append(tmp_loss1)
                eot_loss1 = tf.stack(eot_loss1, 1)
                print('other', eot_loss1.get_shape())
                eot_loss1 = tf.reduce_mean(eot_loss1, axis=1)
                loss1[cur_gpu_idx].append(eot_loss1)
                eot_output = tf.stack(eot_output, 1)
                print('eot_output', eot_output.get_shape())
                eot_output = tf.reduce_mean(eot_output, 1)
                print('eot_output final', eot_output.get_shape())
                self.output.append(eot_output)
                # print('loss1', loss1.get_shape())

                gpu_i += 1

        for i in range(gpu_count):
            with tf.device('/gpu:' + str(i)):
                # print('loss1 before sum', loss1.get_shape())
                loss1[i] = tf.stack(loss1[i], 1)
                loss1[i] = tf.reduce_sum(loss1[i], axis=1)
                loss1[i] = tf.reduce_sum(self.const * loss1[i])

        self.loss1 = tf.add_n(loss1)

        print(self.output[0].shape)
        self.output = tf.stack(self.output, axis=1)
        print(self.output.shape)
        # exit(0)

        # distance to the input data
        self.l2dist = tf.reduce_sum(tf.square(self.newimg - (tf.tanh(self.timg) * self.boxmul + self.boxplus)),
                                    [1, 2, 3])
        self.loss2 = tf.reduce_sum(self.l2dist)

        # compute loss 3
        loss3 = []
        detected = []
        for i in range(gpu_count):
            loss3.append([])
        for name, detector in detector_dict.items():
            cur_gpu_idx = detector_gpu_idx[name] % gpu_count
            with tf.device('/gpu:' + str(cur_gpu_idx)):
                eot_loss3 = []
                for eot_i in range(eot_det_count):
                    tmp_detected = detector.tf_mark(self.newimg, data_format=data_format) - thrs[name]
                    eot_loss3.append(tmp_detected / eot_det_count)
                loss3[cur_gpu_idx].extend(eot_loss3)
        for i in range(gpu_count):
            with tf.device('/gpu:' + str(i)):
                has_det = len(loss3[i]) > 0
                loss3[i] = tf.maximum(0.0, tf.stack(loss3[i], 0))
                loss3[i] = tf.reduce_sum(loss3[i], axis=0)
                if has_det:
                    detected.append(loss3[i])
                loss3[i] = tf.reduce_sum(self.const2 * loss3[i])  # TODO one const per detector

        self.loss3 = tf.add_n(loss3)
        detected = tf.maximum(0.0, tf.stack(detected, 0))
        self.detected = tf.reduce_sum(detected, axis=0)

        # sum up the losses
        self.loss = self.loss1 + self.loss2 + self.loss3

        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)

        tower_grads = []
        for i in range(gpu_count):
            with tf.device('/gpu:' + str(i)):
                grads = optimizer.compute_gradients(loss1[i] + loss3[i], var_list=[modifier])
                tower_grads.append(grads)
        grads = optimizer.compute_gradients(self.loss2, var_list=[modifier])
        tower_grads.append(grads)

        total_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                grads.append(g)
            grad = tf.add_n(grads)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            total_grads.append(grad_and_var)
        self.train = optimizer.apply_gradients(total_grads)
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))
        self.setup.append(self.const2.assign(self.assign_const2))

        self.init = tf.variables_initializer(var_list=[modifier] + new_vars)

    def attack(self, imgs, targets, input_data_format=CHANNELS_LAST, output_data_format=CHANNELS_LAST):
        """
        Perform the L_2 attack on the given images for the given targets.

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
            attack_imgs = imgs[i:i+self.batch_size]
            attack_targets = targets[i:i+self.batch_size]
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
        Run the attack on a batch of images and labels.
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
        imgs = np.arctanh((imgs - self.boxplus) / self.boxmul * 0.999999)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size) * self.initial_const
        CONST2 = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        # the best l2, score, and image attack
        o_bestl2 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = [np.zeros(imgs[0].shape)] * batch_size

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            print(o_bestl2)
            # completely reset adam's internal state.
            self.sess.run(self.init)
            batch = imgs
            batchlab = labs

            bestl2 = [1e10] * batch_size
            bestscore = [-1] * batch_size

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS - 1:
                CONST = upper_bound

            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_timg: batch,
                                       self.assign_tlab: batchlab,
                                       self.assign_const: CONST,
                                       self.assign_const2: CONST2 * 10000})

            prev = 1e6
            for iteration in range(self.MAX_ITERATIONS):

                # perform the attack
                _, l, l2s, scores, det, nimg = self.sess.run([self.train, self.loss,
                                                              self.l2dist, self.output,
                                                              self.detected,
                                                              self.newimg])

                l2s = np.sqrt(l2s) / (2*self.boxmul)
                # print out the losses every 10%
                if iteration % (self.MAX_ITERATIONS // 10) == 0:
                    print(iteration, self.sess.run((self.loss, self.loss1, self.loss2, self.loss3)))

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and iteration % (self.MAX_ITERATIONS // 10) == 0:
                    if l > prev * .9999:
                        break
                    prev = l

                # adjust the best result found so far
                if type(det) == np.float32:
                    det = [0] * batch_size
                for e, (l2, sc, ii, detected) in enumerate(zip(l2s, scores, nimg, det)):
                    # print('have scores',sc)
                    if l2 < bestl2[e] and all([compare(scc, np.argmax(batchlab[e])) for scc in sc]) and detected == 0:
                        bestl2[e] = l2
                        bestscore[e] = np.argmax(sc) % self.num_labels
                    if l2 < o_bestl2[e] and all([compare(scc, np.argmax(batchlab[e])) for scc in sc]) and detected == 0:
                        o_bestl2[e] = l2
                        o_bestscore[e] = np.argmax(sc) % self.num_labels
                        o_bestattack[e] = ii

            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and bestscore[e] != -1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e], CONST[e])
                    CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e], CONST[e])
                    CONST[e] *= self.const_factor
                    if CONST[e] > upper_bound[e]:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        return o_bestattack
