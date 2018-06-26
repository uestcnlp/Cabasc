import tensorflow as tf

def build_softmax(edim, class_num, stddev, inputs, gold_output):
    '''
    build the softmax. 
    wline.shape = [edim, class_num]
    stddev: the random initialize base. 
    input: the softmax basic_layer input.
    gold_output: the gold output.  
    '''
    # the linear basic_layer for softmax.
    wline_softmax = tf.Variable(tf.random_normal([edim, class_num], stddev=stddev), trainable = True)
    bline_softmax = tf.Variable(tf.zeros([1, 1]), trainable = True)
    # shape = [batch_size, class_num]
    softmax_input = tf.add(tf.matmul(inputs, wline_softmax), bline_softmax)
    # the prediction.
    pred = tf.argmax(softmax_input, 1)  # shape = [batch_size, 1]

    # the loss.
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=softmax_input, labels=gold_output)

    return pred, loss
