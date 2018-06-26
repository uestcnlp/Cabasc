import tensorflow as tf
from util.Randomer import Randomer

class SoftmaxCELossLayer(object):
    '''
    the softmax and cross entropy basic_layer.
    '''

    def __init__(self, edim, class_num, stddev=None, params=None):
        '''
        class_num: class type num. 
        edim: the input embedding dim. 
        params = {'wline': wline, 'bline': bline}
        '''
        self.edim = edim
        self.class_num = class_num
        # the linear basic_layer for softmax.
        if params is None:
            self.wline_softmax = tf.Variable(
                Randomer.random_normal(
                    [self.edim, self.class_num]
                ),
                trainable=True
            )
            self.bline_softmax = tf.Variable(tf.zeros([1, 1]), trainable=True)
        else:
            self.wline_softmax = params['wline']
            self.bline_softmax = params['bline']
        
    def forward(self, inputs):
        '''
        input: the softmax basic_layer input.
        input.shape = [batch_size, edim]

        ret:
        pred.shape = [batch_size, 1]
        softmax_input.shape = [batch_size, class_num]
        '''
        # shape = [batch_size, class_num]
        softmax_input = tf.add(tf.matmul(inputs, self.wline_softmax), self.bline_softmax)
        # the prediction.
        pred = tf.argmax(softmax_input, 1)  # shape = [batch_size, 1]

        return pred, softmax_input
    
    def get_loss(self, softmax_input, gold_output):
        '''
        gold_output: the gold output.  
        gold_output.shape = [batch_size, class_num]
        '''
        # the loss.
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=softmax_input, labels=gold_output)
        return loss 
