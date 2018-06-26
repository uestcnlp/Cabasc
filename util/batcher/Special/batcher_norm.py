#coding=utf-8
from util.batcher.Special.BatchData import batch_range
from util.Formater import add_pad
from util.Bitmap import bitmap_by_padid
import numpy as np
import math
import copy


class batcher(object):
    '''
    seq2seqatt batcher.
    '''

    def __init__(
        self,
        samples,
        batch_size,
        class_num,
        max_len,
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

        self.labels = []
        self.texts = []
        self.aspects = []
        self.aspsubs = []
        self.ids = []
        self.eos = eos
        self.batch_size = batch_size
        self.class_num = class_num
        self.pad_idx = pad_idx
        self.pos_id_range = pos_id_range
        self.max_len = max_len

        id2sample = {}
        maxid = -1
        # unpack the samples
        for sample in samples:
            self.unpack_sample(sample)
            id2sample[sample.id] = sample
            if sample.id > maxid:
                maxid = sample.id

        # pad the samples
        samp_lens = len(samples)
        ext_lens = self.batch_size - (samp_lens - samp_lens/self.batch_size * self.batch_size)
        if ext_lens == self.batch_size:
            ext_lens = 0

        ext_ids = np.random.randint(0, maxid + 1, ext_lens)
        for id in ext_ids:
            if id not in id2sample:
                while True:
                    id = np.random.randint(0, maxid + 1)
                    if id in id2sample:
                        break
            self.unpack_sample(id2sample[id])

        if random is True:
            self.rand_idx = np.random.permutation(len(self.texts))
        else:
            self.rand_idx = range(0, len(self.texts))
        self.N = int(math.ceil(float(len(self.texts) / self.batch_size))) + 1
        self.nsamps = len(self.rand_idx)
        self.idx = 0
        self.nN = 0

    def unpack_sample(self, sample):
        self.ids.append(sample.id)
        self.texts.append(sample.text_idxes)
        self.aspects.append(sample.aspect_idxes)
        self.aspsubs.append(sample.aspect_wordpos)
        self.labels.append(sample.label)

    def has_next(self):
        '''
        is hasing next epoch.
        '''
        if self.idx >= self.nsamps:
            return False
        else:
            return True

    def next_batch(self):
        '''
        get the netxt batch_data.
        '''
        self.nN = self.nN + 1
        rins, lab, ret_ids, rinlens, rmaxlens, self.idx, rinlens_float32 = batch_range(
            self.batch_size,
            self.idx,
            self.nsamps,
            self.rand_idx,
            self.class_num,
            self.labels,
            self.ids,
            [self.texts, self.aspsubs, self.aspects]
        )
        fsents = rins[0]
        asubs = rins[1]
        asps = rins[2]
        # context bitmap.
        sent_bitmap = []
        # row sentence lengths.
        sequence_lengs = rinlens[0]
        seq_lens = []
        reverse_lens = []
        for x in xrange(len(sequence_lengs)):
            nl = sequence_lengs[x][0]
            if self.eos:
                nl += 1
            seq_lens.append(nl)  # add the <eos>
            reverse_lens.append(sequence_lengs[x][0])

        # pad index
        add_pad(
            inputs=[fsents],
            max_lens=[self.max_len + 1],
            pad_idx=self.pad_idx
        )
        max_len = self.max_len + 1
        sent_bitmap = bitmap_by_padid(fsents, self.pad_idx)
        alpha_adj = copy.deepcopy(sent_bitmap)
        for row in alpha_adj:
            for i in range(len(row)):
                if row[i] == 1.0:
                    row[i] = 0.0
                else:
                    row[i] = 1.0
                    break
        # count the aspect lens, and size
        # count the memory size
        abs_poses = []
        pos_ids = []
        asp_lens = []
        asp_size = 0
        mem_size = 0
        for x in xrange(len(seq_lens)):
            sl = seq_lens[x]
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
            lest_pid = [self.pos_id_range['oos'] for _ in range(max_len - sl)]
            # build the rets.
            abs_pos = []
            abs_pos.extend(lt_pos)
            abs_pos.extend(asp_pos)
            abs_pos.extend(rt_pos)
            abs_pos.extend(lest_pos)
            abs_poses.append(abs_pos)
            pos_id = []
            pos_id.extend(lt_pid)
            pos_id.extend(asp_pid)
            pos_id.extend(rt_pid)
            pos_id.extend(lest_pid)
            pos_ids.append(pos_id)
        add_pad(
            inputs=[asps],
            max_lens=[asp_size],
            pad_idx=self.pad_idx
        )
        # build the subs.
        asp_subs = []
        f_asp_subs = []
        b_asp_subs = []
        for k in xrange(len(fsents)):
            bias = k * max_len
            asp_sub = []
            asub = asubs[k]
            # test
            # print bias
            # print asub
            # print seq_lens[k]
            # test
            asp_sub = range(bias + asub[0], bias + asub[1])
            f_asp_subs.append(bias + asub[1] - 1)
            b_asp_subs.append(bias + asub[0])

            aslen = len(asp_sub)
            while aslen < asp_size:
                asp_sub.append(bias + max_len - 1)
                aslen = aslen + 1
            asp_subs.append(asp_sub)

        ret_data = {
            'text_idxes' : fsents,
            'batch_ids' : ret_ids,
            'aspect_idxes' : asps,
            'labels' : lab,
            'text_lens' : seq_lens,
            'aspect_lens' : asp_lens,
            'text_reverse_lens' : reverse_lens,
            'aspect_subs' : asp_subs,  #所有经过pad的句子拼接在一起后经过pad的方面在其中的下标,
            'text_bitmap' : sent_bitmap,
            'f_asp_sub' : f_asp_subs,
            'b_asp_sub' : b_asp_subs,
            'alpha_adj' : alpha_adj,
            'abs_poses' : abs_poses,
            'pos_ids' : pos_ids
        }
        return ret_data
