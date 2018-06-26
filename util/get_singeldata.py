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
    aspects.append(sample.aspect_idxes)
    asp_lens.append(len(sample.aspect_idxes))
    aspsubs.append(sample.aspect_wordpos)
    leftsubs.append(range(sample.left_wordpos[1]))
    rightsubs.append(range(len(sample.aspect_idxes),len(sample.right_context_idxes)+len(sample.aspect_idxes)))
    crt_lab = [0.0] * class_num
    crt_lab[sample.label] = 1.0
    labels.append(crt_lab)
    add_pad(
        inputs=[left_ctx,right_ctx],
        max_lens=[left_seq_lens[0]+1,right_seq_lens[0]+1],
        pad_idx=pad_idx
    )
    sent_bitmap = bitmap_by_padid(texts, pad_idx)
    left_sent_bitmap = bitmap_by_padid(left_ctx, pad_idx)
    right_sent_bitmap = bitmap_by_padid(right_ctx, pad_idx)
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
        'aspect_lens':asp_lens ,
        # 'aspect_subs': asp_subs,  # 所有经过pad的句子拼接在一起后经过pad的方面在其中的下标,
        'left_subs': leftsubs,
        'right_subs': rightsubs,
        'text_bitmap': sent_bitmap,
        'left_bitmap': left_sent_bitmap,
        'right_bitmap': right_sent_bitmap
    }
    return ret_data
