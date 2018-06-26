#coding=utf-8
from util.BatchData import batch_range
from util.BatchData import batch_all
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
        random=True,
        pad_idx=0,
        eos=True
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

        # unpack the samples
        for sample in samples:
            self.ids.append(sample.id)
            self.texts.append(sample.text_idxes)
            self.aspects.append(sample.aspect_idxes)
            self.aspsubs.append(sample.aspect_wordpos)
            self.labels.append(sample.label)


        if random is True:
            self.rand_idx = np.random.permutation(len(self.texts))
        else:
            self.rand_idx = range(0, len(self.texts))
        self.N = int(math.ceil(float(len(self.texts) / self.batch_size))) + 1
        self.nsamps = len(self.rand_idx)
        self.idx = 0
        self.nN = 0

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
            max_lens=[rmaxlens[0] + 1],
            pad_idx=self.pad_idx
        )
        max_len = rmaxlens[0] + 1
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

        seq_lens_float32 = []
        for l in seq_lens:
            seq_lens_float32.append([float(l)])

        asp_lens_float32 = []
        for l in asp_lens:
            asp_lens_float32.append([float(l)])

        ret_data = {
            'text_idxes' : fsents,
            'batch_ids' : ret_ids,
            'aspect_idxes' : asps,
            'labels' : lab,
            'text_lens' : seq_lens,
            'text_lens_float32' : seq_lens_float32,
            'aspect_lens' : asp_lens,
            'aspect_lens_float32' : asp_lens_float32,
            'text_reverse_lens' : reverse_lens,
            'aspect_subs' : asp_subs,  #所有经过pad的句子拼接在一起后经过pad的方面在其中的下标,
            'text_bitmap' : sent_bitmap,
            'f_asp_sub' : f_asp_subs,
            'b_asp_sub' : b_asp_subs,
            'alpha_adj' : alpha_adj
        }
        return ret_data
