# coding=utf-8
from util.Bitmap import bitmap_by_padid
from util.Formater import add_pad
import copy
def get_data(sample, class_num,  pad_idx, eos):
    labels = []
    texts = []
    aspects = []
    leftsubs = []
    rightsubs = []
    left_ctx = []
    left_ctx_asp = []
    right_ctx = []
    right_ctx_asp = []
    seq_lens = []
    asp_lens = []
    left_seq_lens =[]
    right_seq_lens = []
    left_ctxasp_lens= []
    right_ctxasp_lens = []
    aspsubs = []
    ids = []

    eos = eos
    class_num = class_num
    pad_idx = pad_idx
    ids.append(sample.id)
    texts.append(sample.text_idxes)
    seq_lens.append(len(sample.text_idxes))
    left_ctx.append(sample.left_context_idxes)
    left_seq_lens.append(len(sample.left_context_idxes))
    tmp = sample.left_context_idxes+sample.aspect_idxes
    tmp.reverse()
    left_ctx_asp.append(tmp)
    right_ctx.append(sample.right_context_idxes)
    right_seq_lens.append(len(sample.right_context_idxes))
    right_ctx_asp.append((sample.aspect_idxes+sample.right_context_idxes))
    left_ctxasp_lens.append(len(left_ctx_asp[0]))
    right_ctxasp_lens.append(len(right_ctx_asp[0]))
    aspects.append(sample.aspect_idxes)
    asp_lens.append(len(sample.aspect_idxes))
    aspsubs.append(sample.aspect_wordpos)
    leftsubs.append(sample.left_wordpos)
    rightsubs.append(sample.aspect_wordpos)
    crt_lab = [0.0] * class_num
    crt_lab[sample.label] = 1.0
    labels.append(crt_lab)
    add_pad(
        inputs=[texts,left_ctx_asp,right_ctx_asp,left_ctx,right_ctx],
        max_lens=[seq_lens[0]+1,seq_lens[0]+1,seq_lens[0]+1,left_seq_lens[0],right_seq_lens[0]],
        pad_idx=pad_idx
    )
    sent_bitmap = bitmap_by_padid(texts, pad_idx)
    left_sent_bitmap = bitmap_by_padid(left_ctx, pad_idx)
    right_sent_bitmap = bitmap_by_padid(right_ctx, pad_idx)
    max_len = seq_lens[0]+1
    left_max_len = left_seq_lens[0]+2
    right_max_len = right_seq_lens[0] + 2
    asp_subs = []
    left_subs = []
    right_subs = []
    left_ngrams=[]
    left_ngram_lens = []
    right_ngrams=[]
    right_ngram_lens = []
    for k in xrange(1):
        bias =  0
        asp_sub = []
        left_sub = []
        right_sub = []
        asub = aspsubs[0]
        lsub = leftsubs[0]
        rsub = rightsubs[0]
        # test
        # print bias
        # print asub
        # print seq_lens[k]
        # test
        asp_sub = range(bias + asub[0], bias + asub[1])
        left_sub = range(bias + lsub[0], bias + lsub[1])
        right_sub = range(bias + rsub[0], bias + rsub[1])


        aslen = len(asp_sub)
        leftlen = len(left_sub)
        rightlen = len(right_sub)

        while leftlen < left_max_len:
            left_sub.append(bias + max_len - 1)
            leftlen = leftlen + 1
        while rightlen < right_max_len:
            right_sub.append(bias + max_len - 1)
            rightlen = rightlen + 1
        left_ngram = []
        left_ngram_len= []
        right_ngram = []
        right_ngram_len = []
        for i in range(1,len(left_sub)):
            left_ngram.append([left_sub[i-1],left_sub[i]])
            if left_sub[i-1] == (max_len - 1)  or left_sub[i] == (max_len - 1):
                left_ngram_len.append(1)
            else:
                left_ngram_len.append(2)
        for i in range(1,len(right_sub)):
            right_ngram.append([right_sub[i-1],right_sub[i]])
            if right_sub[i-1] == (max_len - 1)  or right_sub[i] == (max_len - 1):
                right_ngram_len.append(1)
            else:
                right_ngram_len.append(2)
        asp_subs.append(asp_sub)
        left_subs.append(left_sub)
        right_subs.append(right_sub)
        left_ngrams.append(left_ngram)
        right_ngrams.append(right_ngram)
        left_ngram_lens.append(left_ngram_len)
        right_ngram_lens.append(right_ngram_len)
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
    ret_data={
        'text_idxes': texts,
        'left_ctx_idxes': left_ctx,
        'right_ctx_idxes': right_ctx,
        'left_ctx_asp': left_ctx_asp,
        'right_ctx_asp': right_ctx_asp,
        'batch_ids': ids,
        'aspect_idxes': aspects,
        'labels': labels,
        'text_lens': seq_lens,
        'left_lens': left_seq_lens,
        'right_lens': right_seq_lens,
        'left_ca_lens':left_ctxasp_lens,
        'right_ca_lens':right_ctxasp_lens,
        'left_ngram_lens': left_ngram_lens,
        'right_ngram_lens': right_ngram_lens,
        'aspect_lens':asp_lens ,
        # 'aspect_subs': asp_subs,  # 所有经过pad的句子拼接在一起后经过pad的方面在其中的下标,
        'left_subs': left_subs,
        'right_subs': right_subs,
        'asp_subs' : asp_subs,
        'left_ngrams':left_ngrams,
        'right_ngrams':right_ngrams,
        'text_bitmap': sent_bitmap,
        'left_bitmap': left_sent_bitmap,
        'right_bitmap': right_sent_bitmap,
        'asp_mask':asp_mask
    }
    return ret_data
