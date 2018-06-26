import tensorflow as tf
from util.SoftmaxMask import softmax_mask


class DotAttentionLayer:
    '''
    the dot attention basic_layer.
    '''
    # watt.shape = [2 * edim, 1]
    # batt.shape = [1]

    def __init__(self, edim):
        self.edim = edim

    # context.shape = [batch_size, mem_size, edim]
    # aspect.shape=[batch_size, edim]
    # aspect_3dim.shape = [batch_size, mem_size, edim]
    # ctx_cat_asp.shape = [batch_size, mem_size, 2 * edim]
    # watt_3dim.shape = [batch_size, 2 * edim, 1]
    # gout.shape = [batch_size, mem_size, 1]
    # alpha.shape = [batch_size, mem_size]
    def count_alpha(self, context, aspect, ctx_bitmap):
        '''
        count the content attention (weight)
        '''
        mem_size = tf.shape(context)[1]
        context = context
        aspect = aspect
        # adjust attention
        asp_3dim = tf.reshape(aspect, [-1, self.edim, 1])
        gout = tf.matmul(context, asp_3dim)
        # alpha = tf.nn.softmax(tf.reshape(gout, [-1, mem_size]))
        alpha = softmax_mask(tf.reshape(gout, [-1, mem_size]), ctx_bitmap)
        return alpha

    # context.shape = [batch_size, mem_size, edim]
    # aspect.shape=[batch_size, edim]
    # aspect_3dim.shape = [batch_size, mem_size, edim]
    # ctx_cat_asp.shape = [batch_size, mem_size, 2 * edim]
    # watt_3dim.shape = [batch_size, 2 * edim, 1]
    # gout.shape = [batch_size, mem_size, 1]
    # alpha.shape = [batch_size, mem_size]
    # vec.shape = [batch_size, 1, edim]
    # ctx_bitmap.shape = [batch_size, mem_size]
    def forward(self, context, aspect, ctx_bitmap):
        '''
        count the attention weight and weighted the context embeddings. 
        '''
        mem_size = tf.shape(context)[1]
        context = context
        aspect = aspect
        # adjust attention
        alpha = self.count_alpha(context, aspect, ctx_bitmap)
        vec = tf.matmul(
            tf.reshape(alpha, [-1, 1, mem_size]),
            context
        )
        return vec,alpha

    # context.shape = [batch_size, mem_size, edim]
    # aspect.shape=[batch_size, edim]
    # aspect_3dim.shape = [batch_size, mem_size, edim]
    # ctx_cat_asp.shape = [batch_size, mem_size, 2 * edim]
    # watt_3dim.shape = [batch_size, 2 * edim, 1]
    # gout.shape = [batch_size, mem_size, 1]
    # alpha.shape = [batch_size, mem_size]
    # vec.shape = [batch_size, 1, edim]
    def forward_max_pool(self, context, aspect, ctx_bitmap):
        '''
        count the attention weight and weighted the context embeddings. 
        '''
        mem_size = tf.shape(context)[1]
        context = context
        aspect = aspect
        # adjust attention
        alpha = self.count_alpha(context, aspect, ctx_bitmap)
        vec = tf.reshape(
            tf.reduce_max(
                tf.multiply(
                    tf.tile(
                        tf.reshape(alpha, [-1, mem_size, 1]),
                        [1, 1, self.edim]
                    ),
                    context
                ),
                1
            ),
            [-1, 1, self.edim]
        )
        return vec
