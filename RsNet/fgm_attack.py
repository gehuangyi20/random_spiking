import tensorflow as tf
import numpy as np
from RsNet.dataset_nn import model_mnist_meta
from RsNet.tf_config import CHANNELS_LAST, CHANNELS_FIRST


def optimize_linear(grad, eps, ord=np.inf):
    """
    Solves for the optimal input to a linear function under a norm constraint.
    Optimal_perturbation = argmax_{eta, ||eta||_{ord} < eps} dot(eta, grad)
    :param grad: tf tensor containing a batch of gradients
    :param eps: float scalar specifying size of constraint region
    :param ord: int specifying order of norm
    :returns:
    tf tensor containing optimal perturbation
    """

    # In Python 2, the `list` call in the following line is redundant / harmless.
    # In Python 3, the `list` call is needed to convert the iterator returned by `range` into a list.
    red_ind = list(range(1, len(grad.get_shape())))
    avoid_zero_div = 1e-12
    if ord == np.inf:
        # Take sign of gradient
        optimal_perturbation = tf.sign(grad)
        # The following line should not change the numerical results.
        # It applies only because `optimal_perturbation` is the output of
        # a `sign` op, which has zero derivative anyway.
        # It should not be applied for the other norms, where the
        # perturbation has a non-zero derivative.
        optimal_perturbation = tf.stop_gradient(optimal_perturbation)
    elif ord == 1:
        abs_grad = tf.abs(grad)
        sign = tf.sign(grad)
        max_abs_grad = tf.reduce_max(abs_grad, red_ind, keepdims=True)
        tied_for_max = tf.to_float(tf.equal(abs_grad, max_abs_grad))
        num_ties = tf.reduce_sum(tied_for_max, red_ind, keepdims=True)
        optimal_perturbation = sign * tied_for_max / num_ties
    elif ord == 2:
        square = tf.maximum(avoid_zero_div,
                            tf.reduce_sum(tf.square(grad),
                                          reduction_indices=red_ind,
                                          keepdims=True))
        optimal_perturbation = grad / tf.sqrt(square)
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                  "currently implemented.")

    # Scale perturbation to be the solution for the norm=eps rather than
    # norm=1 problem
    scaled_perturbation = eps * optimal_perturbation

    return scaled_perturbation


class Fgm:
    def __init__(self, model, has_labels=False, eps=0.3, ord=np.inf,
                 clip_min=None, clip_max=None,
                 targeted=False, model_meta=model_mnist_meta, data_format=CHANNELS_LAST):
        """
        TensorFlow implementation of the Fast Gradient Method.
        :param model: the targeted model used for the attack
        :param has_labels: Boolean, whether providing the target label
        :param eps: the epsilon (input variation parameter)
        :param ord: (optional) Order of the norm (mimics NumPy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: Minimum float value for adversarial example components
        :param clip_max: Maximum float value for adversarial example components
        :param targeted: Is the attack targeted or untargeted? Untargeted, the
                         default, will try to make the label incorrect. Targeted
                         will instead try to move in the direction of being more
                         like y.
        :return: a tensor for the adversarial example
        """
        # :param x: the input placeholder
        if data_format == CHANNELS_LAST:
            shape = (None, model_meta.width, model_meta.height, model_meta.channel)
        else:
            shape = (None, model_meta.channel, model_meta.width, model_meta.height)
        self.x = tf.placeholder(tf.float32, shape)
        # :param preds: the model's output tensor (the attack expects the
        #               probabilities, i.e., the output of the softmax)
        self.preds = model.predict(self.x)
        self.has_labels = has_labels
        self.data_format = data_format

        # :param y: (optional) A placeholder for the model labels. If targeted
        #           is true, then provide the target label. Otherwise, only provide
        #           this parameter if you'd like to use true labels when crafting
        #           adversarial samples. Otherwise, model predictions are used as
        #           labels to avoid the "label leaking" effect (explained in this
        #           paper: https://arxiv.org/abs/1611.01236). Default is None.
        #           Labels should be one-hot-encoded.
        if has_labels:
            self.y = tf.placeholder(tf.float32, shape=(None, model_meta.labels))
        else:
            # Using model predictions as ground truth to avoid label leaking
            preds_max = tf.reduce_max(self.preds, 1, keep_dims=True)
            y = tf.to_float(tf.equal(self.preds, preds_max))
            y = tf.stop_gradient(y)
            self.y = y

        y = self.y / tf.reduce_sum(self.y, 1, keep_dims=True)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.preds)
        if targeted:
            loss = -loss

        # Define gradient of loss wrt input
        grad, = tf.gradients(loss, self.x)

        optimal_perturbation = optimize_linear(grad, eps, ord)

        # Add perturbation to original example to obtain adversarial example
        adv_x = self.x + optimal_perturbation

        # If clipping is needed, reset all values outside of [clip_min, clip_max]
        if (clip_min is not None) or (clip_max is not None):
            # We don't currently support one-sided clipping
            assert clip_min is not None and clip_max is not None
            adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

        self.adv_x = adv_x

    def attack(self, imgs, targets, input_data_format=CHANNELS_LAST, output_data_format=CHANNELS_LAST):
        assert input_data_format in (CHANNELS_FIRST, CHANNELS_LAST)
        assert output_data_format in (CHANNELS_FIRST, CHANNELS_LAST)

        # tranpose the input data format to fit the data format of the attack model
        if input_data_format != self.data_format:
            if input_data_format == CHANNELS_LAST:
                # input is channels_last, transpose to channels_first
                imgs = np.transpose(imgs, [0, 3, 1, 2])
            else:
                # input is channels_first, transpose to channels_last
                imgs = np.transpose(imgs, [0, 2, 3, 1])

        if self.has_labels:
            feed_dict = {self.x: imgs, self.y: targets}
        else:
            feed_dict = {self.x: imgs}

        output = self.adv_x.eval(feed_dict=feed_dict)

        # tranpose the output data format of the attack model to fit the output data format
        if output_data_format != self.data_format:
            if output_data_format == CHANNELS_LAST:
                # attack model output is channels_first, transpose to channels_last
                output = np.transpose(output, [0, 2, 3, 1])
            else:
                # attack model output is channels_last, transpose to channels_first
                output = np.transpose(output, [0, 3, 1, 2])

        return output
