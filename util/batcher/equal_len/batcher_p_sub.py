#coding=utf-8
from util.BatchData import batch_range
from util.BatchData import batch_all
from util.Formater import add_pad
from util.Bitmap import bitmap_by_padid
import numpy as np
import math
import copy
import random


class batcher(object):
    '''
    seq2seqatt batcher.
    '''

    def __init__(
        self,
        samples,
        class_num,
        random=True,
        pad_idx=0,
        eos=True,
        pos_id_range={  #pos_ids: 82,81,...,2,1,0,0,0,83,84,...,164,165
            'asp':0,
            'left_start':1,
            'left_end':82,
            'right_start':83,
            'right_end':165,
            'oos':166 # out of sentence
        }
    ):
        '''
        the init funciton.
        '''
        self.eos = eos
        self.batch_size = 0
        self.class_num = class_num
        self.pad_idx = pad_idx
        self.pos_id_range = pos_id_range
        self.len_dic={}
        # unpack the samples
        for sample in samples:
            len_key = len(sample.text_idxes)
            if self.len_dic.has_key(len_key):
                self.len_dic[len_key].append(sample)
            else:
                self.len_dic[len_key] =[sample]
            # self.ids.append(sample.id)
            # self.texts.append(sample.text_idxes)
            # self.left_ctx.append(sample.left_context_idxes)
            # left_tmp = sample.left_context_idxes + sample.aspect_idxes
            # left_tmp.reverse()
            # self.left_ctx_asp.append(left_tmp)
            # right_tmp = sample.aspect_idxes+sample.right_context_idxes
            # self.right_ctx_asp.append(right_tmp)
            # self.right_ctx.append(sample.right_context_idxes)
            # self.aspects.append(sample.aspect_idxes)
            # self.aspsubs.append(sample.aspect_wordpos)
            # self.left_aspsubs.append([0,(len(sample.aspect_idxes))])
            # self.right_aspsubs.append([0,(len(sample.aspect_idxes))])
            # self.leftsubs.append(sample.left_wordpos)
            # self.rightsubs.append(sample.right_wordpos)
            # self.labels.append(sample.label)

        self.key_list = self.len_dic.keys()
        if random is True:
            self.rand_idx = np.random.permutation(len(self.key_list))
        else:
            self.rand_idx = range(0, len(self.key_list))
        self.idx = 0

    def has_next(self):
        '''
        is hasing next epoch. 
        '''
        if self.idx >= len(self.rand_idx):
            return False
        else:
            return True

    def next_batch(self):
        '''
        get the netxt batch_data.
        '''
        self.labels = []
        self.texts = []
        self.aspects = []
        self.leftsubs = []
        self.rightsubs = []
        self.left_ctx = []
        self.left_ctx_asp = []
        self.right_ctx = []
        self.right_ctx_asp = []
        self.aspsubs = []
        self.left_aspsubs = []
        self.right_aspsubs = []
        self.ids = []
        samplelist = self.len_dic[self.key_list[self.rand_idx[self.idx]]]
        random.shuffle(samplelist)
        for sample in samplelist:
            self.ids.append(sample.id)
            self.texts.append(sample.text_idxes)
            self.left_ctx.append(sample.left_context_idxes)
            left_tmp = sample.left_context_idxes + sample.aspect_idxes
            left_tmp.reverse()
            self.left_ctx_asp.append(left_tmp)
            right_tmp = sample.aspect_idxes+sample.right_context_idxes
            self.right_ctx_asp.append(right_tmp)
            self.right_ctx.append(sample.right_context_idxes)
            self.aspects.append(sample.aspect_idxes)
            self.aspsubs.append(sample.aspect_wordpos)
            self.left_aspsubs.append([0,(len(sample.aspect_idxes))])
            self.right_aspsubs.append([0,(len(sample.aspect_idxes))])
            self.leftsubs.append(sample.left_wordpos)
            self.rightsubs.append(sample.right_wordpos)
            self.labels.append(sample.label)

        rins, lab, rinlens, rmaxlens, rinlens_float32 = batch_all(
            [self.texts, self.aspsubs, self.aspects, self.left_ctx, self.right_ctx, self.leftsubs, self.rightsubs,
             self.left_ctx_asp, self.right_ctx_asp, self.left_aspsubs, self.right_aspsubs],
            self.labels,
            self.class_num,
        )
        self.idx += 1
        fsents = rins[0]
        asubs = rins[1]
        asps = rins[2]
        left_ctx = rins[3]
        right_ctx = rins[4]
        lsubs = rins[5]
        rsubs = rins[6]
        l_ctx_asp = rins[7]
        r_ctx_asp = rins[8]
        l_asubs = rins[9]
        r_asubs = rins[10]

        # context bitmap.
        sent_bitmap = []
        # row sentence lengths.
        sequence_lengs = rinlens[0]
        left_sequence_lengs = rinlens[3]
        right_sequence_lengs = rinlens[4]
        l_ctx_asp_len =rinlens[7]
        r_ctx_asp_len = rinlens[8]
        seq_lens = []
        left_seq_lens = []
        right_seq_lens = []
        l_ca_len = []
        r_ca_len = []
        reverse_lens = []
        for x in xrange(len(sequence_lengs)):
            nl = sequence_lengs[x][0]
            if self.eos:
                nl += 1
            seq_lens.append(nl)  # add the <eos>
            reverse_lens.append(sequence_lengs[x][0])
            left_seq_lens.append(left_sequence_lengs[x][0])
            right_seq_lens.append(right_sequence_lengs[x][0])
            l_ca_len.append(l_ctx_asp_len[x][0])
            r_ca_len.append(r_ctx_asp_len[x][0])


        # for x in xrange(len(left_sequence_lengs)):
        #     nl = left_sequence_lengs[x][0]
        #     left_seq_lens.append(nl)
        #     # reverse_lens.append(sequence_lengs[x][0])
        # for x in xrange(len(right_sequence_lengs)):
        #     nl = right_sequence_lengs[x][0]
        #     right_seq_lens.append(nl)
        #     # reverse_lens.append(sequence_lengs[x][0])
        left_max_len = rmaxlens[3]
        right_max_len = rmaxlens[4]
        # pad index
        add_pad(
            inputs=[fsents, left_ctx, right_ctx,l_ctx_asp,r_ctx_asp],
            max_lens=[rmaxlens[0]+1, rmaxlens[0], rmaxlens[0],rmaxlens[0]+1,rmaxlens[0]+1],
            pad_idx=self.pad_idx
        )
        max_len = rmaxlens[0]+1
        sent_bitmap = bitmap_by_padid(fsents, self.pad_idx)
        left_sent_bitmap = bitmap_by_padid(left_ctx, self.pad_idx)
        right_sent_bitmap = bitmap_by_padid(right_ctx, self.pad_idx)
        alpha_adj = copy.deepcopy(sent_bitmap)
        for row in alpha_adj:
            for i in range(len(row)):
                if row[i] == 1.0:
                    row[i] = 0.0
                else:
                    row[i] = 1.0
                    break

        left_alpha_adj = copy.deepcopy(left_sent_bitmap)
        for row in left_alpha_adj:
            for i in range(len(row)):
                if row[i] == 1.0:
                    row[i] = 0.0
                else:
                    row[i] = 1.0
                    break
        right_alpha_adj = copy.deepcopy(right_sent_bitmap)
        for row in right_alpha_adj:
            for i in range(len(row)):
                if row[i] == 1.0:
                    row[i] = 0.0
                else:
                    row[i] = 1.0
                    break
        # count the aspect lens, and size
        # count the memory size

        abs_poses = []
        left_abs_poses = []
        right_abs_poses = []

        pos_ids = []
        left_pos_ids = []
        right_pos_ids = []

        asp_lens = []
        asp_size = 0
        mem_size = 0
        for x in xrange(len(seq_lens)):
            sl = seq_lens[x]
            left_l = left_seq_lens[x]
            right_l = right_seq_lens[x]
            asub = asubs[x]
            al = asub[1] - asub[0]
            asp_lens.append(al)
            if al > asp_size:
                asp_size = al
            ms = sl - al
            if ms > mem_size:
                mem_size = ms
            # count the position
            # left
            lt_pos = range(asub[0] + 1)[1:]
            lt_pid = range(asub[0] + 1)[1:]
            lt_pos.reverse()
            lt_pid.reverse()
            left_start = self.pos_id_range['left_start']
            tmp_lt_pid = np.array(lt_pid)
            tmp_lt_pid += left_start - 1
            lt_pid = tmp_lt_pid.tolist()
            # right
            rt_pos = range(sl - asub[1] + 1)[1:]
            rt_pid = range(sl - asub[1] + 1)[1:]
            right_start = self.pos_id_range['right_start']
            tmp_rt_pid = np.array(rt_pid)
            tmp_rt_pid += right_start - 1
            rt_pid = tmp_rt_pid.tolist()            
            # aspect
            asp_pos = [0 for _ in range(al)]
            asp_pid = [self.pos_id_range['asp'] for _ in range(al)]
            # lest. the pads.
            lest_pos = [0 for _ in range(max_len - sl)]
            l_lest_pos = [0 for _ in range(rmaxlens[3] - left_l)]
            r_lest_pos = [0 for _ in range(rmaxlens[4] - right_l)]
            lest_pid = [self.pos_id_range['oos'] for _ in range(max_len - sl)]
            l_lest_pid = [self.pos_id_range['oos'] for _ in range(rmaxlens[3] - left_l)]
            r_lest_pid = [self.pos_id_range['oos'] for _ in range(rmaxlens[4] - right_l)]
            # build the rets.
            abs_pos = []
            abs_pos.extend(lt_pos)
            abs_pos.extend(asp_pos)
            abs_pos.extend(rt_pos)
            abs_pos.extend(lest_pos)
            abs_poses.append(abs_pos)
            lt_pos.extend(l_lest_pos)
            left_abs_poses.append(lt_pos)
            rt_pos.extend(r_lest_pos)
            right_abs_poses.append(rt_pos)
            pos_id = []
            pos_id.extend(lt_pid)
            pos_id.extend(asp_pid)
            pos_id.extend(rt_pid)
            pos_id.extend(lest_pid)
            pos_ids.append(pos_id)
            lt_pid.extend(l_lest_pid)
            left_pos_ids.append(lt_pid)
            rt_pid.extend(r_lest_pid)
            right_pos_ids.append(rt_pid)
        add_pad(
            inputs=[asps],
            max_lens=[asp_size],
            pad_idx=self.pad_idx
        )
        asp_mask = []
        for x in range(len(seq_lens)):
            asp_mask.append([])
        for i in range(len(seq_lens)):
            for x in range(left_seq_lens[i]):
                asp_mask[i].append(1.0)
            for x in range(asp_lens[i]):
                asp_mask[i].append(0.5)
            for x in range(right_seq_lens[i]):
                asp_mask[i].append(1.0)
            asp_mask[i].append(0.0)

        asp_pos = []
        for x in range(len(seq_lens)):
            asp_pos.append([])
        for i in range(len(seq_lens)):
            for x in range(left_seq_lens[i]):
                asp_pos[i].append(0)
            for x in range(asp_lens[i]):
                asp_pos[i].append(1)
            for x in range(right_seq_lens[i]):
                asp_pos[i].append(0)

        left_a_mask=[]
        for x in range(len(seq_lens)):
            left_a_mask.append([])
        for i in range(len(seq_lens)):
            for x in range(asp_lens[i]):
                left_a_mask[i].append(1)
            for x in range(seq_lens[i] - asp_lens[i]):
                left_a_mask[i].append(0)

        left_mask = []
        for x in range(len(seq_lens)):
            left_mask.append([])
        for i in range(len(seq_lens)):
            for x in range(asp_lens[i]):
                left_mask[i].append(0)
            for x in range(seq_lens[i] - asp_lens[i]):
                left_mask[i].append(1)

        left_mask2 = []
        for x in range(len(seq_lens)):
            left_mask2.append([])
        for i in range(len(seq_lens)):
            for x in range(left_seq_lens[i]):
                left_mask2[i].append(1)
            for x in range(seq_lens[i] - left_seq_lens[i]):
                left_mask2[i].append(0)

        right_mask2 = []
        for x in range(len(seq_lens)):
            right_mask2.append([])
        for i in range(len(seq_lens)):
            for x in range(right_seq_lens[i]):
                right_mask2[i].append(1)
            for x in range(seq_lens[i] - right_seq_lens[i]):
                right_mask2[i].append(0)

        left_asp_mask =[]
        for x in range(len(seq_lens)):
            left_asp_mask.append([])
        for i in range(len(seq_lens)):
            for x in range(l_ca_len[i]):
                left_asp_mask[i].append(1)
            for x in range(seq_lens[i] - l_ca_len[i]):
                left_asp_mask[i].append(0)


        right_asp_mask = []
        for x in range(len(seq_lens)):
            right_asp_mask.append([])
        for i in range(len(seq_lens)):
            for x in range(r_ca_len[i]):
                right_asp_mask[i].append(1)
            for x in range(seq_lens[i] - r_ca_len[i]):
                right_asp_mask[i].append(0)

        # build the subs.
        asp_subs = []
        left_subs = []
        right_subs = []
        f_asp_subs = []
        b_asp_subs = []
        f_left_subs = []
        b_left_subs = []
        f_right_subs = []
        b_right_subs = []
        window_subs = []
        window_lens = []
        for k in xrange(len(fsents)):
            bias = k * max_len
            asp_sub = []
            left_sub = []
            right_sub = []
            asub = asubs[k]
            lsub = lsubs[k]
            rsub = rsubs[k]
            # test
            # print bias
            # print asub
            # print seq_lens[k]
            # test
            asp_sub = range(bias + asub[0], bias + asub[1])
            left_sub = range(bias+lsub[0],bias + lsub[1])
            right_sub = range(bias + rsub[0], bias + rsub[1])

            f_asp_subs.append(bias + asub[1] - 1)
            b_asp_subs.append(bias + asub[0])
            f_left_subs.append(bias + lsub[1] -1)
            b_left_subs.append(bias + lsub[0])
            f_right_subs.append(bias + rsub[1] - 1)
            b_right_subs.append(bias + rsub[0])
            window_sub = []
            window_len = []
            window_size = 5
            w = []
            for x in range(len(fsents[k])):
                lenth = 0
                for s in range(1,window_size):
                    if x - s<0:
                        left_1 = bias + max_len - 1
                    else:
                        left_1 = bias + fsents[k][x - 2]
                        lenth += 1
                    w.append(left_1)
                # if x -1 < 0 :
                #     left_2 = bias + max_len - 1
                # else:
                #     left_2 = bias + fsents[k][x - 1]
                #     lenth += 1
                for s in range(1, window_size):
                    if x + s >= max_len:
                        right_1 = bias + max_len - 1
                    else:
                        right_1 = bias + fsents[k][x + 1]
                        lenth += 1
                    w.append(right_1)
                # if x + 2 >= max_len:
                #     right_2 = bias + max_len - 1
                # else:
                #     right_2 = bias + fsents[k][x + 2]
                #     lenth += 1
                if lenth == 0:
                    lenth +=1
                window_len.append(lenth)
                window_sub.append(w)
            window_subs.append(window_sub)
            window_lens.append(window_len)
            aslen = len(asp_sub)
            leftlen = len(left_sub)
            rightlen = len(right_sub)
            while aslen < asp_size:
                asp_sub.append(bias + max_len - 1)
                aslen = aslen + 1
            while leftlen < left_max_len:
                left_sub.append(bias + max_len - 1)
                leftlen = leftlen + 1
            while rightlen < right_max_len:
                right_sub.append(bias + max_len - 1)
                rightlen = rightlen + 1
            asp_subs.append(asp_sub)
            left_subs.append(left_sub)
            right_subs.append(right_sub)

        ret_data = {
            'text_idxes' : fsents,
            'left_ctx_idxes': left_ctx,
            'right_ctx_idxes': right_ctx,
            'left_ctx_asp': l_ctx_asp,
            'right_ctx_asp':r_ctx_asp,
            'batch_ids' : self.ids,
            'aspect_idxes' : asps,
            'labels' : lab,
            'text_lens' : seq_lens,
            'left_lens': left_seq_lens,
            'right_lens': right_seq_lens,
            'left_ca_lens':l_ca_len,
            'right_ca_lens':r_ca_len,
            'aspect_lens' : asp_lens,
            'text_reverse_lens' : reverse_lens,
            'aspect_subs' : asp_subs,  #所有经过pad的句子拼接在一起后经过pad的方面在其中的下标,
            'window_subs' : window_subs,
            'window_lens' : window_lens,
            # 'left_subs' : left_subs,
            # 'right_subs' : right_subs,
            'text_bitmap' : sent_bitmap,
            'left_bitmap': left_sent_bitmap,
            'right_bitmap': right_sent_bitmap,
            # 'f_asp_sub' : f_asp_subs,
            # 'b_asp_sub' : b_asp_subs,
            # 'f_left_sub': f_left_subs,
            # 'b_left_sub': b_left_subs,
            # 'f_right_sub': f_right_subs,
            # 'b_right_sub': b_right_subs,
            'alpha_adj' : alpha_adj,
            'abs_poses' : abs_poses,
            'left_abs_poses' : left_abs_poses,
            'right_abs_poses' : right_abs_poses,
            'pos_ids' : pos_ids,
            'left_pos_ids' : left_pos_ids,
            'right_pos_ids' : right_pos_ids,
            'asp_mask': asp_mask,
            'left_a_mask': left_a_mask,
            'left_mask': left_mask,
            'left_mask2':left_mask2,
            'right_mask2': right_mask2,
            'left_asp_mask': left_asp_mask,
            'right_asp_mask': right_asp_mask,
            'asp_pos': asp_pos
        }
        return ret_data
