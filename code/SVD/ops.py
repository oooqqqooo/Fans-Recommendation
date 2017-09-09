import tensorflow as tf


def inference_svd(user_batch, item_batch, user_num, item_num, dim=5, device="/cpu:0"):
    with tf.device("/cpu:0"):
        #tf.get_variable_scope().reuse == True

        bias_global_new111 = tf.get_variable("bias_global_new111", shape=[])
        w_bias_user_new111 = tf.get_variable("embd_bias_user_new111", shape=[user_num])
        w_bias_item_new111 = tf.get_variable("embd_bias_item_new111", shape=[item_num])
        bias_user_new111 = tf.nn.embedding_lookup(w_bias_user_new111, user_batch, name="bias_user_new111")
        bias_item_new111 = tf.nn.embedding_lookup(w_bias_item_new111, item_batch, name="bias_item_new111")
        w_user_new111= tf.get_variable("embd_user_new111", shape=[user_num, dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        w_item_new111 = tf.get_variable("embd_item_new111", shape=[item_num, dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        embd_user_new111 = tf.nn.embedding_lookup(w_user_new111, user_batch, name="embedding_user111")
        embd_item_new111 = tf.nn.embedding_lookup(w_item_new111, item_batch, name="embedding_item111")
    
    with tf.device(device):
        infer = tf.reduce_sum(tf.multiply(embd_user_new111, embd_item_new111), 1)
        infer = tf.add(infer, bias_global_new111)
        infer = tf.add(infer, bias_user_new111)
        infer = tf.add(infer, bias_item_new111, name="svd_inference")
        regularizer = tf.add(tf.nn.l2_loss(embd_user_new111), tf.nn.l2_loss(embd_item_new111), name="svd_regularizer")
    return infer, regularizer


def optimization(infer, regularizer, rate_batch, learning_rate=0.001, reg=0.1, device="/cpu:0"):
    global_step = tf.train.get_global_step()
    assert global_step is not None
    with tf.device(device):
        cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))
        penalty = tf.constant(reg, dtype=tf.float32, shape=[], name="l2")
        cost = tf.add(cost_l2, tf.multiply(regularizer, penalty))
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
    return cost, train_op
