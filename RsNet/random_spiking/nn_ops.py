from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
import tensorflow as tf
import numbers


def random_perturb(x, rand_prob, _min, _max, axis=None, seed=None, name=None):
    with ops.name_scope(name, "random_perturb", [x]) as name:
        x = ops.convert_to_tensor(x, name="x")
        if not x.dtype.is_floating:
            raise ValueError("x has to be a floating point tensor since it's going to"
                             " be scaled. Got a %s tensor instead." % x.dtype)
        if isinstance(rand_prob, numbers.Real) and not 0 <= rand_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                             "range (0, 1], got %g" % rand_prob)
        if isinstance(min, numbers.Real) and isinstance(max, numbers.Real) and min > max:
            raise ValueError("min must be equal or less than max, get min %g and max %g "
                             % (_min, _max))

        rand_prob = ops.convert_to_tensor(rand_prob,
                                          dtype=x.dtype,
                                          name="keep_prob")
        rand_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        _min = ops.convert_to_tensor(_min,
                                     dtype=x.dtype,
                                     name="keep_prob")
        _min.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        _max = ops.convert_to_tensor(_max,
                                     dtype=x.dtype,
                                     name="keep_prob")
        _max.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        # Do nothing if we know rand_prob == 0
        if tensor_util.constant_value(rand_prob) == 0:
            return x

        x_shape = array_ops.shape(x)
        rank_x = len(x.get_shape())
        scale = array_ops.ones(rank_x, dtype=tf.int32)

        # determine the smallest element as perturbation group
        # for example: tensor shape (3,3,3) [[[1,3], [4,5]], [[6,8], [9,0]]]
        # given axis 2, them element [1,3] will either remain original, or changed
        # altogether
        if axis is None:
            axis = rank_x

        noise_shape = array_ops.concat([x_shape[:axis], scale[axis:]], 0)
        scale_shape = array_ops.concat([scale[:axis], x_shape[axis:]], 0)

        # uniform [rand_prob, 1.0 + rand_prob)
        random_tensor = rand_prob
        random_tensor += random_ops.random_uniform(noise_shape,
                                                   seed=seed,
                                                   dtype=x.dtype)
        # 1. if [rand_prob, 1.0) and 0. if [1.0, 1.0 + rand_prob)
        drop_tensor = array_ops.tile(math_ops.floor(random_tensor), scale_shape)

        # 0 if [rand_prob, 1.0) and 1. if [1.0, 1.0 + rand_prob)
        keep_tensor = 1 - drop_tensor
        random_perturb_tensor = random_ops.random_uniform(x_shape, seed=seed, dtype=x.dtype)
        noise_tensor = _min + (_max - _min) * random_perturb_tensor

        ret = x * keep_tensor + noise_tensor * drop_tensor

        return ret


def random_sample_top(x, axis, sample_rate, top_percent, seed=None, name=None):
    with ops.name_scope(name, "random_sample_top", [x]) as name:
        x = ops.convert_to_tensor(x, name="x")

        if not x.dtype.is_floating:
            raise ValueError("x has to be a floating point tensor since it's going to"
                             " be scaled. Got a %s tensor instead." % x.dtype)
        if isinstance(top_percent, numbers.Real) and not 0 <= top_percent <= 1:
            raise ValueError("percent must be a scalar tensor or a float in the "
                             "range (0, 1], got %g" % top_percent)
        if isinstance(sample_rate, numbers.Real) and not 0 <= sample_rate <= 1:
            raise ValueError("sample_rate must be a scalar tensor or a float in the "
                             "range (0, 1], got %g" % sample_rate)

        sample_rate = ops.convert_to_tensor(sample_rate,
                                            dtype=x.dtype,
                                            name="sample_rate")

        top_percent = ops.convert_to_tensor(top_percent,
                                            dtype=x.dtype,
                                            name="top_percent")
        if tensor_util.constant_value(sample_rate) == 0 or tensor_util.constant_value(top_percent) == 0:
            return x

        shape = x.get_shape().as_list()
        rank = len(shape)
        axis = sorted(axis, reverse=True)

        # build first permutation index array, suppose x has shape [B, H, W, C]
        # axis is [3], then we transpose the x to [C, B, H, W].

        trans_idx_0 = list(range(rank))
        for i in axis:
            del trans_idx_0[i]

        tf_shape = array_ops.shape(x)
        flat_dim = tf.constant(1)
        for val in trans_idx_0:
            flat_dim *= tf_shape[val]

        trans_idx_0 = axis + trans_idx_0
        x = array_ops.transpose(x, trans_idx_0)
        # X1= x

        # After that, for each channel, we find the top_percent value among
        # [B, H, W]. So we get max_value and top_percent_value for that channel
        # Sample sample_rate% element in the channel [B, H, W]. For those elements,
        # We random assign value between max_value and top_percent_value

        x_shape = array_ops.shape(x)

        # get top_k
        x = array_ops.reshape(x, [-1, flat_dim])
        x_shape_top_k = array_ops.shape(x)

        k = math_ops.floor(tf.cast(flat_dim, tf.float32) * top_percent)
        k = math_ops.maximum(k, 1)
        top_k_val, topk_idx = nn_ops.top_k(x, tf.cast(k, tf.int32))
        top_k_val_order = array_ops.transpose(top_k_val)
        _max = array_ops.expand_dims(top_k_val_order[0], 1)
        _max = array_ops.tile(_max, [1, flat_dim])
        _min = array_ops.expand_dims(top_k_val_order[tf.cast(k-1, tf.int32)], 1)
        _min = array_ops.tile(_min, [1, flat_dim])

        # uniform [rand_prob, 1.0 + rand_prob)
        random_tensor = sample_rate
        random_tensor += random_ops.random_uniform(x_shape_top_k,
                                                   seed=seed,
                                                   dtype=x.dtype)
        # 1. if [rand_prob, 1.0) and 0. if [1.0, 1.0 + rand_prob)
        drop_tensor = math_ops.floor(random_tensor)

        # 0 if [rand_prob, 1.0) and 1. if [1.0, 1.0 + rand_prob)
        keep_tensor = 1 - drop_tensor
        random_perturb_tensor = random_ops.random_uniform(x_shape_top_k, seed=seed, dtype=x.dtype)
        noise_tensor = _min + (_max - _min) * random_perturb_tensor
        x = x * keep_tensor + noise_tensor * drop_tensor

        # reshape back to original
        x = array_ops.reshape(x, x_shape)

        # we build the permutation index array to transpose the array back to original
        # dimension order
        #
        # We find the index of element of trans_idx_0 in sorted order
        trans_idx_1 = sorted(range(rank), key=lambda _k: trans_idx_0[_k])
        x = array_ops.transpose(x, trans_idx_1)

        #print(shape, rank, trans_idx_0, trans_idx_1)

        #return X1, x, _max, _min
        return x


def random_max_pool(x, scale_rate, seed=None, name=None):
    with ops.name_scope(name, "random_max_pool", [x]) as name:
        x = ops.convert_to_tensor(x, name="x")

        if not x.dtype.is_floating:
            raise ValueError("x has to be a floating point tensor since it's going to"
                             " be scaled. Got a %s tensor instead." % x.dtype)

        if isinstance(scale_rate, numbers.Real) and not 0 <= scale_rate:
            raise ValueError("sample_rate must be a scalar tensor or a float >= 0"
                             ", got %g" % scale_rate)

        scale_rate = ops.convert_to_tensor(scale_rate,
                                           dtype=x.dtype,
                                           name="sample_rate")

        shape = x.get_shape().as_list()
        dimension = shape[-3]
        rank = len(shape)

        # build first permutation index array, suppose x has shape [B, H, W, C]
        # axis is [3], then we transpose the x to [C, B, H, W].
        trans_idx_0 = [rank-1] + list(range(rank-1))

        # compute number of matrix
        tf_shape = array_ops.shape(x)
        mat_dim = tf.constant(1)
        for val in trans_idx_0[:-2]:
            mat_dim *= tf_shape[val]

        x = array_ops.transpose(x, trans_idx_0)
        inter_x_shape = array_ops.shape(x)
        # x1 = x

        # build dialog matrix
        mat_diag = array_ops.diag(array_ops.ones(dimension, dtype=tf.float32))
        mat_diag_expand = array_ops.expand_dims(mat_diag, 0)
        mat_diag_tile = array_ops.tile(mat_diag_expand, [mat_dim, 1, 1])

        # build random spike matrix
        tr_range = math_ops.range(dimension, dtype=tf.float32)
        tr_range_expand = array_ops.expand_dims(tr_range, 0)

        # we random sample from [0, dimension-1) which is the index offset to the index of the dialog element
        # in the matrix. Then, we convert the offset to the real index value
        mat_rand_tensor = random_ops.random_uniform(inter_x_shape[:-1], seed=seed, dtype=x.dtype)
        mat_rand_idx_offset = math_ops.ceil(mat_rand_tensor * (dimension-1))
        mat_rand_idx_base = array_ops.tile(tr_range_expand, [mat_dim, 1])
        mat_rand_idx_base = array_ops.reshape(mat_rand_idx_base, inter_x_shape[:-1])
        mat_rand_idx = (mat_rand_idx_base + mat_rand_idx_offset) % dimension

        # expand the index to matrix. Each element in the row has the same value which is the original index value.
        mat_rand_tile_shape = array_ops.concat([array_ops.ones(rank-1, dtype=tf.int32), [dimension]], 0)
        mat_shape = array_ops.concat([inter_x_shape[:-1], [dimension]], 0)
        mat_rand_idx_final = array_ops.tile(array_ops.expand_dims(mat_rand_idx, -1), mat_rand_tile_shape)

        tr_range_tile = array_ops.tile(tr_range_expand, [mat_dim*dimension, 1])
        mat_rand_base = array_ops.reshape(tr_range_tile, mat_shape)

        mat_rand_final = math_ops.cast(math_ops.equal(mat_rand_idx_final, mat_rand_base), tf.float32)

        mat_factor = array_ops.reshape(mat_diag_tile, mat_shape)
        mat_factor += mat_rand_final * scale_rate

        x = math_ops.matmul(mat_factor, x)
        # x2 = x

        # we build the permutation index array to transpose the array back to original
        # dimension order
        #
        # We find the index of element of trans_idx_0 in sorted order
        trans_idx_1 = sorted(range(rank), key=lambda _k: trans_idx_0[_k])
        x = array_ops.transpose(x, trans_idx_1)

        return x


def random_pool_k(x, k, seed=None, name=None):
    with ops.name_scope(name, "random_pool_k", [x]) as name:
        x = ops.convert_to_tensor(x, name="x")

        if not x.dtype.is_floating:
            raise ValueError("x has to be a floating point tensor since it's going to"
                             " be scaled. Got a %s tensor instead." % x.dtype)

        if isinstance(k, numbers.Integral) and not 0 <= k:
            raise ValueError("sample_rate must be a integer tensor or a integer >= 0"
                             ", got %g" % k)

        k = ops.convert_to_tensor(k, dtype=tf.float32, name="sample_k")
        k = math_ops.cast(k, dtype=tf.int32)

        shape = x.get_shape().as_list()
        dimension = shape[-3]
        print(dimension, shape[0], shape[-1], shape[-2], shape[-3], shape)
        rank = len(shape)

        # build first permutation index array, suppose x has shape [B, H, W, C]
        # axis is [3], then we transpose the x to [C, B, H, W].
        trans_idx_0 = [rank - 1] + list(range(rank - 1))

        # compute number of matrix
        tf_shape = array_ops.shape(x)
        mat_dim = tf.constant(1)
        for val in trans_idx_0[:-2]:
            mat_dim *= tf_shape[val]

        x = array_ops.transpose(x, trans_idx_0)
        inter_x_shape = array_ops.shape(x)

        # build dialog matrix
        mat_diag = array_ops.diag(array_ops.ones(dimension, dtype=tf.float32))
        mat_diag_expand = array_ops.expand_dims(mat_diag, 0)
        mat_diag_tile = array_ops.tile(mat_diag_expand, [mat_dim, 1, 1])

        # build random spike matrix
        tr_range = math_ops.range(dimension, dtype=tf.int32)
        tr_range_expand = array_ops.expand_dims(tr_range, 0)

        # multiplication matrix shape
        mat_shape = array_ops.concat([inter_x_shape[:-1], [dimension]], 0)

        # we random value in the matrix having shape mat_shape. For each row (last dimension)
        # we choose the top k value. The index of these top k value will be the index offset
        # of the index of the dialog element
        # in the matrix. Then, we convert the offset to the real index value
        mat_rand_tensor_shape = array_ops.concat([inter_x_shape[:-1], [dimension-1]], 0)
        mat_rand_tensor = random_ops.random_uniform(mat_rand_tensor_shape, seed=seed, dtype=x.dtype)
        mat_rand_offset_k = nn_ops.top_k(mat_rand_tensor, k).indices + 1

        # compute the real index by adding the diagonal index to the offset
        mat_rand_idx_base_shape = array_ops.concat([inter_x_shape[:-1], [k]], 0)
        mat_rand_idx_base = array_ops.tile(
            array_ops.reshape(tr_range, [1, dimension, 1]),
            [mat_dim, 1, k])
        mat_rand_idx_base = array_ops.reshape(mat_rand_idx_base, mat_rand_idx_base_shape)
        mat_rand_idx = (mat_rand_idx_base + mat_rand_offset_k) % dimension

        mat_rand_idx_expand_shape = array_ops.concat(
            [array_ops.ones(rank, dtype=tf.int32), [dimension]], 0)
        mat_rand_idx_expand = array_ops.expand_dims(mat_rand_idx, -1)
        mat_rand_idx_expand = array_ops.tile(mat_rand_idx_expand, mat_rand_idx_expand_shape)
        mat_rand_idx_base_expand = array_ops.tile(
            array_ops.reshape(tr_range, [1, 1, dimension]),
            [mat_dim, k, dimension])
        mat_rand_idx_base_expand = array_ops.reshape(
            mat_rand_idx_base_expand, array_ops.shape(mat_rand_idx_expand))
        mat_rand_idx_bool_expand = math_ops.cast(
            math_ops.equal(mat_rand_idx_expand, mat_rand_idx_base_expand), tf.float32)

        mat_rand_trans_idx = list(range(rank - 1)) + [rank, rank - 1]
        mat_rand_idx_bool_trans = array_ops.transpose(mat_rand_idx_bool_expand, mat_rand_trans_idx)
        mat_rand_idx_final = array_ops.reshape(
            math_ops.reduce_sum(mat_rand_idx_bool_trans, -1), mat_shape)

        mat_factor = array_ops.reshape(mat_diag_tile, mat_shape)
        mat_factor += mat_rand_idx_final

        x = math_ops.matmul(mat_factor, x)
        # we build the permutation index array to transpose the array back to original
        # dimension order
        #
        # We find the index of element of trans_idx_0 in sorted order
        trans_idx_1 = sorted(range(rank), key=lambda _k: trans_idx_0[_k])
        x = array_ops.transpose(x, trans_idx_1)

        return x


def random_spike_vector(x, scale_rate, seed=None, name=None):
    with ops.name_scope(name, "random_spike_vector", [x]) as name:
        x = ops.convert_to_tensor(x, name="x")
        if not x.dtype.is_floating:
            raise ValueError("x has to be a floating point tensor since it's going to"
                             " be scaled. Got a %s tensor instead." % x.dtype)

        if isinstance(scale_rate, numbers.Real) and not 0 <= scale_rate:
            raise ValueError("sample_rate must be a scalar tensor or a float >= 0"
                             ", got %g" % scale_rate)

        scale_rate = ops.convert_to_tensor(scale_rate,
                                           dtype=x.dtype,
                                           name="sample_rate")

        # Do nothing if we know rand_prob == 0
        if tensor_util.constant_value(scale_rate) == 0:
            return x

        shape = x.get_shape().as_list()
        dimension = shape[-1]

        # compute number of matrix
        x_shape = array_ops.shape(x)
        mat_dim = math_ops.reduce_prod(x_shape[:-1])
        # for val in trans_idx_0[:-2]:
        #     mat_dim *= tf_shape[val]
        #
        # build dialog matrix
        mat_diag = array_ops.diag(array_ops.ones(dimension, dtype=tf.float32))
        mat_diag_expand = array_ops.expand_dims(mat_diag, 0)
        mat_diag_final = array_ops.tile(mat_diag_expand, [mat_dim, 1, 1])

        x_flat = array_ops.reshape(x, [mat_dim, dimension])
        x_flat_expend = array_ops.expand_dims(x_flat, -1)
        x_flat_diag = array_ops.tile(x_flat_expend, [1, 1, dimension]) * mat_diag_final

        # build random spike matrix
        tr_range = math_ops.range(dimension, dtype=tf.float32)
        tr_range_expand = array_ops.expand_dims(tr_range, 0)

        # we random sample from [0, dimension-1) which is the index offset to the index of the dialog element
        # in the matrix. Then, we convert the offset to the real index value
        mat_rand_tensor = random_ops.random_uniform([mat_dim, dimension], seed=seed, dtype=x.dtype)
        mat_rand_idx_offset = math_ops.ceil(mat_rand_tensor * (dimension - 1))
        mat_rand_idx_base = array_ops.tile(tr_range_expand, [mat_dim, 1])
        mat_rand_idx = (mat_rand_idx_base + mat_rand_idx_offset) % dimension
        mat_rand_idx_final = array_ops.tile(array_ops.expand_dims(mat_rand_idx, -1), [1, 1, dimension])

        tr_range_tile = array_ops.tile(tr_range_expand, [mat_dim * dimension, 1])
        mat_rand_base = array_ops.reshape(tr_range_tile, [mat_dim, dimension, dimension])

        mat_rand_final = math_ops.cast(math_ops.equal(mat_rand_idx_final, mat_rand_base), tf.float32)

        mat_factor = mat_diag_final + mat_rand_final * scale_rate

        x_flat_spike = math_ops.matmul(mat_factor, x_flat_diag)
        x_sum_spike = math_ops.reduce_sum(x_flat_spike, -1)
        x_spike = array_ops.reshape(x_sum_spike, x_shape)

        return x_spike


def random_spike_vector_k(x, k, seed=None, name=None):
    with ops.name_scope(name, "random_spike_vector", [x]) as name:
        x = ops.convert_to_tensor(x, name="x")
        if not x.dtype.is_floating:
            raise ValueError("x has to be a floating point tensor since it's going to"
                             " be scaled. Got a %s tensor instead." % x.dtype)

        if isinstance(k, numbers.Integral) and not 0 <= k:
            raise ValueError("sample_rate must be a integer tensor or a integer >= 0"
                             ", got %g" % k)

        k = ops.convert_to_tensor(k, dtype=tf.float32, name="sample_k")
        k = math_ops.cast(k, dtype=tf.int32)

        # Do nothing if we know rand_prob == 0
        if tensor_util.constant_value(k) == 0:
            return x

        shape = x.get_shape().as_list()
        dimension = shape[-1]

        # compute number of matrix
        x_shape = array_ops.shape(x)
        mat_dim = math_ops.reduce_prod(x_shape[:-1])

        x_flat = array_ops.reshape(x, [mat_dim, dimension])

        # we random sample from [0, dimension) which is the index of the  element in the vector.
        mat_rand_tensor_k = random_ops.random_uniform([mat_dim * k, dimension, 1], seed=seed, dtype=x.dtype)
        mat_rand_idx_k = math_ops.cast(math_ops.floor(mat_rand_tensor_k * dimension), tf.int32)

        # build the the vector index which we want to gather
        x_vector_idx_range = array_ops.reshape(math_ops.range(mat_dim, dtype=tf.int32), [mat_dim, 1])
        x_vector_idx_range_expand = array_ops.reshape(
            array_ops.tile(x_vector_idx_range, [1, k*dimension]), [mat_dim * k, dimension, 1])

        # concat vector index and element index, then gather the random spike vectors
        x_spike_add_idx = array_ops.concat([x_vector_idx_range_expand, mat_rand_idx_k], axis=2)
        x_spike_add_expand = array_ops.gather_nd(x_flat, x_spike_add_idx)

        # sum up the random spiking
        x_spike_addition_independ = array_ops.reshape(x_spike_add_expand, [mat_dim, k, dimension])
        x_spike_addition_independ_tran = array_ops.transpose(x_spike_addition_independ, [0, 2, 1])
        x_spike_addition = math_ops.reduce_sum(x_spike_addition_independ_tran, axis=-1)
        # prevent compute gradients in back propagation
        x_spike_addition = array_ops.stop_gradient(x_spike_addition)

        x_spike = array_ops.reshape(x_spike_addition + x_flat, x_shape)

        return x_spike


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

        _max = math_ops.reduce_max(x)
        _min = math_ops.reduce_min(x)
        random_perturb_tensor = random_ops.random_uniform(x_shape, seed=seed, dtype=x.dtype)
        noise_tensor = _min + (_max - _min) * random_perturb_tensor
        x_noise = noise_tensor * drop_tensor
        x_noise = array_ops.stop_gradient(x_noise)
        ret = x_keep + x_noise

        return ret


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

        _max = array_ops.reshape(math_ops.reduce_max(x, list(range(1, shape_len))), exp_shape)
        _min = array_ops.reshape(math_ops.reduce_min(x, list(range(1, shape_len))), exp_shape)
        random_perturb_tensor = random_ops.random_uniform(x_shape, seed=seed, dtype=x.dtype)
        noise_tensor = _min + (_max - _min) * random_perturb_tensor
        x_noise = noise_tensor * drop_tensor
        x_noise = array_ops.stop_gradient(x_noise)
        ret = x_keep + x_noise

        return ret
