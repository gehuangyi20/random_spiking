from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
import numbers


# random spiking layer implementation used in the paper, random value is
# sampled between the min and max output of all neuron output in batch-wised
def random_spike_sample_scaling(x, sample_rate, scaling, seed=None, name=None):
    with ops.name_scope(name, "random_sample_top", [x]) as name:
        x = ops.convert_to_tensor(x, name="x")

        if not x.dtype.is_floating:
            raise ValueError("x has to be a floating point tensor since it's going to"
                             " be scaled. Got a %s tensor instead." % x.dtype)
        if isinstance(scaling, numbers.Real) and not 0 <= scaling:
            raise ValueError("scaling must be a scalar tensor or a non-negative float, got %g" % scaling)
        if isinstance(sample_rate, numbers.Real) and not 0 <= sample_rate <= 1:
            raise ValueError("sample_rate must be a scalar tensor or a float in the "
                             "range (0, 1], got %g" % sample_rate)

        sample_rate = ops.convert_to_tensor(sample_rate,
                                            dtype=x.dtype,
                                            name="sample_rate")

        scaling = ops.convert_to_tensor(scaling,
                                        dtype=x.dtype,
                                        name="top_percent")
        if tensor_util.constant_value(sample_rate) == 0 or tensor_util.constant_value(scaling) == 0:
            return x

        x_shape = array_ops.shape(x)

        # uniform [rand_prob, 1.0 + rand_prob)
        random_tensor = sample_rate
        random_tensor += random_ops.random_uniform(x_shape, seed=seed, dtype=x.dtype)

        # 0. if [rand_prob, 1.0) and 1. if [1.0, 1.0 + rand_prob)
        drop_tensor = math_ops.floor(random_tensor)
        # 1 if [rand_prob, 1.0) and 0. if [1.0, 1.0 + rand_prob)
        keep_tensor = 1 - drop_tensor

        x_keep = x * keep_tensor

        # min and max batch-wised
        _max = math_ops.reduce_max(x)
        _min = math_ops.reduce_min(x)
        random_perturb_tensor = random_ops.random_uniform(x_shape, seed=seed, dtype=x.dtype)
        noise_tensor = _min + (_max - _min) * random_perturb_tensor
        x_noise = noise_tensor * drop_tensor
        x_noise = array_ops.stop_gradient(x_noise)
        ret = x_keep + x_noise

        return ret


# similar to the pervious random spiking layer implementation.
# The random value is generated in sample-wised instead of batch-wised.
def random_spike_sample_scaling_per_sample(x, sample_rate, scaling, seed=None, name=None):
    with ops.name_scope(name, "random_sample_top", [x]) as name:
        x = ops.convert_to_tensor(x, name="x")

        if not x.dtype.is_floating:
            raise ValueError("x has to be a floating point tensor since it's going to"
                             " be scaled. Got a %s tensor instead." % x.dtype)
        if isinstance(scaling, numbers.Real) and not 0 <= scaling:
            raise ValueError("scaling must be a scalar tensor or a non-negative float, got %g" % scaling)
        if isinstance(sample_rate, numbers.Real) and not 0 <= sample_rate <= 1:
            raise ValueError("sample_rate must be a scalar tensor or a float in the "
                             "range (0, 1], got %g" % sample_rate)

        sample_rate = ops.convert_to_tensor(sample_rate,
                                            dtype=x.dtype,
                                            name="sample_rate")

        scaling = ops.convert_to_tensor(scaling,
                                        dtype=x.dtype,
                                        name="top_percent")
        if tensor_util.constant_value(sample_rate) == 0 or tensor_util.constant_value(scaling) == 0:
            return x

        x_shape = array_ops.shape(x)
        shape = x.get_shape().as_list()
        shape_len = len(shape)
        exp_shape = array_ops.concat([[x_shape[0]], [1]*(shape_len-1)], 0)

        # uniform [rand_prob, 1.0 + rand_prob)
        random_tensor = sample_rate
        random_tensor += random_ops.random_uniform(x_shape, seed=seed, dtype=x.dtype)

        # 0. if [rand_prob, 1.0) and 1. if [1.0, 1.0 + rand_prob)
        drop_tensor = math_ops.floor(random_tensor)
        # 1 if [rand_prob, 1.0) and 0. if [1.0, 1.0 + rand_prob)
        keep_tensor = 1 - drop_tensor

        x_keep = x * keep_tensor

        # min and max sample-wised
        _max = array_ops.reshape(math_ops.reduce_max(x, list(range(1, shape_len))), exp_shape)
        _min = array_ops.reshape(math_ops.reduce_min(x, list(range(1, shape_len))), exp_shape)
        random_perturb_tensor = random_ops.random_uniform(x_shape, seed=seed, dtype=x.dtype)
        noise_tensor = _min + (_max - _min) * random_perturb_tensor
        x_noise = noise_tensor * drop_tensor
        x_noise = array_ops.stop_gradient(x_noise)
        ret = x_keep + x_noise

        return ret
