#coding=utf-8
import numpy as np
import random
import tensorflow as tf
import copy
from util.get_singeldata import get_data
from NN import NN
from basic_layer.LinearLayer_3dim import LinearLayer_3dim
from basic_layer.SoftmaxCELossLayer import SoftmaxCELossLayer
from layer.MLPAtt_tri import mlpatt
from model.basic_layer.LSTMLayer import LSTMLayer
from util.AccCalculater import cau_samples_acc
from util.Pooler import pooler
from util.Printer import TIPrint
from util.batcher.equal_len.batcher_p import batcher
import time

class MLPAttNN(NN):
    """
    The memory network with context attention. 
    该模型是：
    使用含有attention层的MLP,输入为left_memory,right_memory, aspect
    gate(left,right)
    rnn计算position
    """
    # ctx_input.shape=[batch_size, mem_size]

    def __init__(self, config):
        super(MLPAttNN, self).__init__(config)
        self.config = None
        if config != None:
            self.config = config
            # the config.
            self.datas = config['dataset']
            self.nepoch = config['nepoch']  # the train epoches.
            self.batch_size = config['batch_size']  # the max train batch size.
            self.init_lr = config['init_lr']  # the initialize learning rate.
            # the base of the initialization of the parameters.
            self.stddev = config['stddev']
            self.edim = config['edim']  # the dim of the embedding.
            self.max_grad_norm = config['max_grad_norm']   # the L2 norm.
            # the pad id in the embedding dictionary.
            self.pad_idx = config['pad_idx']
            # the pre-train embedding.
            # shape = [nwords, edim]
            self.pre_embedding = config['pre_embedding']
            # generate the pre_embedding mask.
            self.pre_embedding_mask = np.ones(np.shape(self.pre_embedding))
            self.pre_embedding_mask[self.pad_idx] = 0
            # the class number = len([positive, negative, neutral]).
            self.class_num = config['class_num']
            # update the pre-train embedding or not.
            self.emb_up = config['emb_up']
            # is update the learning rate with the process.
            self.is_update_lr = config['update_lr']
            # the active function.
            self.active = config['active']
            # is reversing the input
            self.reverse = config['reverse']
            # multi basic_layer
            self.multi_layer = config['multi_layer']
            # is using the same lstm cell
            self.same_cell = config['same_cell']
            # hidden size
            self.hidden_size = config['hidden_size']
            # should add <eos>
            self.eos = config['eos']
            # 解码阶段，attention layer的输出是否与上一个时间步FwAtt-GRU的输出相加
            self.is_add = config['is_add']
            # 若is_add=True，那么是否加一次
            self.is_add_onetime = config['is_add_onetime']
            # decoder阶段循环次数
            self.time_step = config['time_step']
            # 采用那种rnn unit.
            self.cell = config['cell']
            self.w_shapes = config['w_shapes']
            self.att = config['attlayer']
            self.use_gate = config['use_gate']
            self.is_share_att = config['is_share_att']
            self.is_concat_asp = config['is_concat_asp']
            self.print_all = config['print_all']
            # the aspect phrase embeddings' pool method.
            if self.is_update_lr:
                self.decay_steps = config['decay_steps']
                self.decay_rate = config['decay_rate']

        # the input.
        self.inputs = None
        self.left_inputs = None
        self.right_inputs = None
        self.aspects = None
        # sequence length
        self.sequence_length = None
        self.left_length = None
        self.right_length = None
        self.reverse_length = None
        self.aspect_length = None
        # the label input. (on-hot, the true label is 1.)
        self.lab_input = None
        self.embe_dict = None  # the embedding dictionary.
        # the optimize set.
        self.global_step = None  # the step counter.
        self.loss = None  # the loss of one batch evaluate.
        self.lr = None  # the learning rate.
        self.optimizer = None  # the optimiver.
        self.optimize = None  # the optimize action.
        # the mask of pre_train embedding.
        self.pe_mask = None
        # the predict.
        self.pred = None
        # the params need to be trained.
        self.params = None
        # the tensorflow session.
        # the position embedding.
        self.pos_emb_dict = np.random.normal(
            0,
            config['emb_stddev'],
            [167, self.edim]
        )
        # pos_id_range={  #pos_ids: 82,81,...,2,1,0,0,0,83,84,...,164,165
        #     'asp':0,
        #     'left_start':1,
        #     'left_end':82,
        #     'right_start':83,
        #     'right_end':165,
        #     'oos':166 # out of sentence
        # }

    def build_model(self):
        '''
        build the MemNN model
        '''
        # the input.
        self.inputs = tf.placeholder(
            tf.int32,
            [None, None],
            name="inputs"
        )
        self.left_inputs = tf.placeholder(
            tf.int32,
            [None, None],
            name="left_inputs"
        )
        self.right_inputs = tf.placeholder(
            tf.int32,
            [None, None],
            name="right_inputs"
        )
        self.left_ctx_asp = tf.placeholder(
            tf.int32,
            [None, None],
            name="left_ctx_asp"
        )
        self.right_ctx_asp = tf.placeholder(
            tf.int32,
            [None, None],
            name="right_ctx_asp"
        )
        self.sequence_length = tf.placeholder(
            tf.int64,
            [None],
            name='sequence_length'
        )
        self.left_length = tf.placeholder(
            tf.int64,
            [None],
            name='left_length'
        )
        self.right_length = tf.placeholder(
            tf.int64,
            [None],
            name='right_length'
        )
        self.sent_bitmap = tf.placeholder(
            tf.float32,
            [None, None],
            name="sent_bitmap"
        )
        # self.left_sent_bitmap = tf.placeholder(
        #     tf.float32,
        #     [None, None],
        #     name="left_sent_bitmap"
        # )
        # self.right_sent_bitmap = tf.placeholder(
        #     tf.float32,
        #     [None, None],
        #     name="right_sent_bitmap"
        # )
        if self.reverse is True:
            self.reverse_length = tf.placeholder(
                tf.int64,
                [None],
                name='reverse_length'
            )
        self.aspects = tf.placeholder(
            tf.int32,
            [None, None],
            name="aspects"
        )
        self.aspect_length = tf.placeholder(
            tf.int64,
            [None],
            name="aspect_length"
        )
        # shape = [batch_size, class_num]
        self.lab_input = tf.placeholder(
            tf.float32,
            [None, self.class_num],
            name="lab_input"
        )
        # self.left_subs = tf.placeholder(
        #     tf.int32,
        #     [None, None],
        #     name="left_subs"
        # )
        # self.right_subs = tf.placeholder(
        #     tf.int32,
        #     [None, None],
        #     name="right_subs"
        # )

        # the position ids.
        # self.pos_ids = tf.placeholder(
        #     tf.int64,
        #     [None, None],
        #     name="pos_ids"
        # )

        # the lookup dict.
        self.embe_dict = tf.Variable(
            self.pre_embedding,
            dtype=tf.float32,
            trainable=self.emb_up
        )

        # the pre_train embedding mask
        self.pe_mask = tf.Variable(
            self.pre_embedding_mask,
            dtype=tf.float32,
            trainable=False
        )
        self.asp_mask = tf.placeholder(
            tf.float32,
            [None, None],
            name="asp_mask"
        )
        self.embe_dict *= self.pe_mask
        # lookup basic_layer.
        # the sentence embedding matrix.
        grulayer1 = LSTMLayer(
            self.hidden_size,
            cell=self.cell
        )
        linear1=LinearLayer_3dim(
            [self.hidden_size,1],
            self.stddev
        )
        grulayer2 = LSTMLayer(
            self.hidden_size,
            cell=self.cell,
        )
        linear2 = LinearLayer_3dim(
            [self.hidden_size, 1],
            self.stddev
        )
        inputs = tf.nn.embedding_lookup(self.embe_dict, self.inputs)
        org_memory = inputs
        aspects = tf.nn.embedding_lookup(self.embe_dict, self.aspects)
        batch_size = tf.shape(org_memory)[0]
        encoded_aspect = pooler(
            aspects,
            'mean',
            axis=1,
            sequence_length=tf.cast(tf.reshape(self.aspect_length, [batch_size, 1]), tf.float32)
        )
        encoded_aspect = tf.reshape(encoded_aspect, [batch_size, self.hidden_size])
        # build org_memory
        # left_inputs = tf.nn.embedding_lookup(self.embe_dict, self.left_inputs)
        # right_inputs = tf.nn.embedding_lookup(self.embe_dict, self.right_inputs)
        left_inputs_asp = tf.nn.embedding_lookup(self.embe_dict,self.left_ctx_asp)
        right_inputs_asp = tf.nn.embedding_lookup(self.embe_dict,self.right_ctx_asp)

        last_output_left,_ =grulayer1.forward(left_inputs_asp, sequence_length=self.left_length,last_outputs=False,name='left')
        last_output_right,_ = grulayer2.forward(right_inputs_asp, sequence_length=self.right_length, last_outputs=False, name='right')

        last_output_left = tf.sigmoid(linear1.forward(last_output_left))
        last_output_left = tf.reverse_sequence(last_output_left,self.left_length,1,0)

        last_output_right = tf.sigmoid(linear2.forward(last_output_right))
        last_output_right = tf.reverse_sequence(last_output_right, self.right_length, 1, 0)
        last_output_right = tf.reverse(last_output_right,[1])
        location = last_output_left+last_output_right
        asp_mask = tf.reshape(self.asp_mask,[batch_size,-1,1])
        location = location*asp_mask
        self.location = tf.reshape(location,[batch_size,1,-1])
        location = tf.tile(location,[1,1,self.hidden_size])
        # left_location = tf.nn.embedding_lookup(tf.reshape(last_output_left,[-1,1]),self.left_subs)
        # right_location = tf.nn.embedding_lookup(tf.reshape(last_output_right,[-1,1]),self.right_subs)

        # left_location = tf.tile(left_location,[1,1,self.hidden_size])
        # aspects_location = tf.zeros([1,tf.shape(aspects)[1],self.hidden_size],tf.float32)
        # left_location = tf.concat([left_location,aspects_location],1)
        # right_location = tf.tile(right_location,[1,1,self.hidden_size])
        # location = tf.concat([left_location, right_location], 1)

        org_memory = org_memory*location
        # build org_memory
        # the aspect embedding matrix.

        last_output = pooler(
            org_memory,
            'mean',
            axis=1,
            sequence_length=tf.cast(tf.reshape(self.sequence_length, [batch_size, 1]), tf.float32)
        )
        last_output = tf.reshape(last_output, [batch_size, self.hidden_size])
        mlphops = mlpatt(
            self.hidden_size,
            self.w_shapes,
            self.stddev,
            self.use_gate,
            self.is_share_att,
            att_func=self.att
        )
        last_output, om_alpha= mlphops.mlp_compute_tri(
            org_memory,
            encoded_aspect,
            last_output,
            self.sent_bitmap,
            self.is_concat_asp,
            self.is_add
        )
        self.last_output = last_output
        self.om_alpha = om_alpha
        # build the softmax
        softmaxce = SoftmaxCELossLayer(
            edim=self.hidden_size,
            class_num=self.class_num,
            stddev=self.stddev
        )

        self.pred, softmax_input = softmaxce.forward(
            inputs=last_output
        )

        self.loss = softmaxce.get_loss(
            softmax_input=softmax_input,
            gold_output=self.lab_input
        )

        # the optimize.
        self.params = tf.trainable_variables()
        self.optimize = super(MLPAttNN, self).optimize_normal(
            self.loss, self.params)

    def train(self, sess, train_data, test_data=None, saver=None, threshold_acc=0.999):
        '''
        Train the mocel. 
        The data in the train_data, test_data:
            data = [contexts, aspects, labels, positons,
               rowtexts, rowaspects, fullsents, subpositions]
        '''
        max_acc = 0.0
        max_train_acc = 0.0
        for _ in xrange(self.nepoch):   # epoch round.
            cost = 0.0  # the cost of each epoch.
            bt = batcher(
                samples=train_data.samples,
                class_num=self.class_num,
                random=True,
                pad_idx=self.pad_idx,
                eos=self.eos
            )
            while bt.has_next():    # batch round.
                # get this batch data
                batch_data = bt.next_batch()
                # build the feed_dict
                feed_dict = {
                    self.inputs: batch_data['text_idxes'],
                    self.left_inputs: batch_data['left_ctx_idxes'],
                    self.right_inputs: batch_data['right_ctx_idxes'],
                    self.aspects: batch_data['aspect_idxes'],
                    self.sequence_length: batch_data['text_lens'],
                    self.left_length: batch_data['left_ca_lens'],
                    self.right_length: batch_data['right_ca_lens'],
                    self.aspect_length: batch_data['aspect_lens'],
                    self.lab_input: batch_data['labels'],
                    self.sent_bitmap: batch_data['text_bitmap'],
                    self.left_ctx_asp: batch_data['left_ctx_asp'],
                    self.right_ctx_asp: batch_data['right_ctx_asp'],
                    self.asp_mask:batch_data['asp_mask']
                    # self.left_sent_bitmap: batch_data['left_bitmap'],
                    # self.right_sent_bitmap: batch_data['right_bitmap'],
                    # self.pos_ids: batch_data['pos_ids']
                }
                # if self.reverse is True:
                #     feed_dict[self.reverse_length] = batch_data['text_reverse_lens']
                # for keys in feed_dict.keys():
                #     print str(keys) + str(feed_dict[keys])
            # samples = train_data.samples
            # id2samples = train_data.id2sample
            # rinids = range(len(samples))
            # random.shuffle(rinids)
            # for id in rinids:
            #     sample = copy.deepcopy(id2samples[id])
            #     ret = get_data(sample,class_num=self.class_num,pad_idx=self.pad_idx,eos=self.eos)
            #     feed_dict = {
            #         self.inputs: ret['text_idxes'],
            #         # self.left_inputs: ret['left_ctx_idxes'],
            #         # self.right_inputs: ret['right_ctx_idxes'],
            #         self.aspects: ret['aspect_idxes'],
            #         self.sequence_length: ret['text_lens'],
            #         self.left_length: ret['left_lens'],
            #         self.right_length: ret['right_lens'],
            #         self.aspect_length: ret['aspect_lens'],
            #         self.lab_input: ret['labels'],
            #         self.sent_bitmap: ret['text_bitmap'],
            #         # self.left_sent_bitmap: ret['left_bitmap'],
            #         # self.right_sent_bitmap: ret['right_bitmap'],
            #         self.left_subs: ret['left_subs'],
            #         self.right_subs: ret['right_subs'],
            #         self.left_ctx_asp: ret['left_ctx_asp'],
            #         self.right_ctx_asp: ret['right_ctx_asp']
            #             }
                # train
                crt_loss, crt_step, opt = sess.run(
                    [self.loss, self.global_step, self.optimize],
                    feed_dict=feed_dict
                )
                cost += np.sum(crt_loss)
            train_acc = self.test(sess, train_data)
            print "train epoch: " + str(_) + \
                "    cost: " + \
                str(cost / len(train_data.samples)) + \
                "    acc: " + \
                str(train_acc) + \
                "    crt_step:" + \
                str(crt_step / 28)
            if test_data != None:
                test_acc = self.test(sess, test_data)
                print "                  test_acc: " + str(test_acc)
                if max_acc < test_acc:
                    max_acc = test_acc
                    max_train_acc = train_acc
                    test_data.update_best()
                    if max_acc > threshold_acc:
                        self.save_model(sess, self.config, saver)
                print "                   max_acc: " + str(max_acc)

        if max_acc > threshold_acc:

            if self.print_all:
                suf = TIPrint(test_data.samples, self.config, True,
                        {'predict_accuracy': max_acc, 'train_accurayc': max_train_acc}, True)
                TIPrint(test_data.samples, self.config, False,
                        {'predict_accuracy': max_acc, 'train_accurayc': max_train_acc}, True, suf)
            else:
                TIPrint(test_data.samples, self.config, False,
                        {'predict_accuracy': max_acc, 'train_accurayc': max_train_acc}, True)

        return max_acc



    def test(self, sess, test_data):
        '''
        the test function.
        test_data = [contexts, aspects, labels, positons]

        contexts: shape = [len(samples), None], the test samples' context, 
        the len(each sample context) is not fixed, the fixed version is mem_size. 

        aspects: shape = [len(samples), None], 
        the test samples' aspect, the len(each sample aspect) is not fixed.

        labels: shape = [len(samples)]

        positons.shape = [len(samples), 2], the aspect's positon in the sample, 
        include from and to (where the 2 means).

        the model input include the apsect lens.
        '''

        # batch data.
        preds = []
        alphas = []
        outputs = []
        bt = batcher(
            samples=test_data.samples,
            class_num=self.class_num,
            random=True,
            pad_idx=self.pad_idx,
            eos=self.eos
        )
        while bt.has_next():  # batch round.
            # get this batch data
            batch_data = bt.next_batch()
            # build the feed_dict
            feed_dict = {
                self.inputs: batch_data['text_idxes'],
                self.left_inputs: batch_data['left_ctx_idxes'],
                self.right_inputs: batch_data['right_ctx_idxes'],
                self.aspects: batch_data['aspect_idxes'],
                self.sequence_length: batch_data['text_lens'],
                self.left_length: batch_data['left_ca_lens'],
                self.right_length: batch_data['right_ca_lens'],
                self.aspect_length: batch_data['aspect_lens'],
                self.lab_input: batch_data['labels'],
                self.sent_bitmap: batch_data['text_bitmap'],
                self.left_ctx_asp: batch_data['left_ctx_asp'],
                self.right_ctx_asp: batch_data['right_ctx_asp'],
                self.asp_mask: batch_data['asp_mask']
                # self.left_sent_bitmap: batch_data['left_bitmap'],
                # self.right_sent_bitmap: batch_data['right_bitmap'],
                # self.pos_ids: batch_data['pos_ids']
            }

            if self.reverse is True:
                feed_dict[self.reverse_length] = batch_data['text_reverse_lens']
        #
        # samples = test_data.samples
        # id2samples = test_data.id2sample
        # ids = range(len(samples))
        # rinids = ids
        # for id in rinids:
        #     sample = copy.deepcopy(id2samples[id])
        #     ret = get_data(sample, class_num=self.class_num, pad_idx=self.pad_idx, eos=self.eos)
        #     feed_dict = {
        #         self.inputs: ret['text_idxes'],
        #         # self.left_inputs: ret['left_ctx_idxes'],
        #         # self.right_inputs: ret['right_ctx_idxes'],
        #         self.aspects: ret['aspect_idxes'],
        #         self.sequence_length: ret['text_lens'],
        #         self.left_length: ret['left_lens'],
        #         self.right_length: ret['right_lens'],
        #         self.aspect_length: ret['aspect_lens'],
        #         self.lab_input: ret['labels'],
        #         self.sent_bitmap: ret['text_bitmap'],
        #         # self.left_sent_bitmap: ret['left_bitmap'],
        #         # self.right_sent_bitmap: ret['right_bitmap'],
        #         self.left_subs: ret['left_subs'],
        #         self.right_subs: ret['right_subs'],
        #         self.left_ctx_asp: ret['left_ctx_asp'],
        #         self.right_ctx_asp: ret['right_ctx_asp']
        #     }

            # test
            pred, om_tmpalphas,loc_tmp,out_tmp= sess.run(
                [self.pred, self.om_alpha,self.location,self.last_output],
                feed_dict=feed_dict
            )
            test_data.pack_preds(pred, batch_data['batch_ids'])
            # em_tmpalphas = test_data.transform_ext_matrix(em_tmpalphas)
            om_tmpalphas = test_data.transform_ext_matrix(om_tmpalphas)
            test_data.pack_ext_matrix('om', om_tmpalphas, batch_data['batch_ids'])
            test_data.pack_ext_matrix('loc', loc_tmp, batch_data['batch_ids'])
            #test_data.pack_ext_matrix('output', out_tmp, batch_data['batch_ids'])

            # test_data.pack_ext_matrix('right', em_tmpalphas, ret['batch_ids'])

            # if alphas == []:
            #     for i in xrange(len(tmpalphas)):
            #         alphas.append(tmpalphas[i].tolist())
            #         outputs.append(tmpoutputs[i].tolist())
            # else:
            #     for i in xrange(len(tmpalphas)):
            #         alphas[i].extend(tmpalphas[i].tolist())
            #         outputs[i].extend(tmpoutputs[i].tolist())
        # calculate the acc
        acc = cau_samples_acc(test_data.samples)
        return acc

# def load_model(self): # first doing the __init__() does, then build
# model, last reset the w and b.

# def dump_model(self):
