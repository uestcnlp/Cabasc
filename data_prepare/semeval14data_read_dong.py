import nltk
import xml.etree.ElementTree as ET
from data_prepare.entity.sample import Sample
from data_prepare.entity.samplepack import Samplepack
import copy
import numpy as np
from util.FileDumpLoad import dump_file
# import sys, os
# reload(sys)
# sys.setdefaultencoding('utf-8')

def load_data(train_file, test_file, pad_idx=0, class_num = 3):
    '''
    ret = [contexts, aspects, labels, positions] ,
    context.shape = [len(samples), None], None should be the len(context); 
    aspects.shape = [len(samples), None], None should be the len(aspect);
    labels.shape = [len(samples)]
    positions.shape = [len(samples), 2], the 2 means from and to.
    '''
    # the global param.
    word2idx = {}  # the ret
    word2idx['<pad>'] = pad_idx
    idx_cnt = 0
    # load the data
    train_data, idx_cnt = _load_data(train_file, word2idx, idx_cnt, pad_idx, class_num)
    test_data, idx_cnt = _load_data(test_file, word2idx, idx_cnt, pad_idx, class_num)
    return train_data, test_data, word2idx


def _load_data(file_path, word2idx, idx_cnt, pad_idx ,class_num):
    # tree = ET.parse(file_path)
    # root = tree.getroot()
    samplepack = Samplepack()
    samples = []
    now_id = 0
    retdata = []  # the ret
    contexts = []  # the ret
    fullsents = []  # the full sentences.
    aspects = []  # the ret
    labels = []  # the ret
    positons = []  # the ret
    subpositions = []  # the subscript positions.
    rowtexts = []  # the row texts
    rowaspects = []  # the row aspects
    data=open(file_path,'r')
    tmp=[]
    for line in data:
        tmp.append(line)
    for i in xrange(0,len(tmp),3):
        sample = Sample()
        row_text = tmp[i].lower().strip()
        # left=row_text.split('$t$')[0]
        # row=row_text.split()[1]
        index1=row_text.index('$t$')


        # rmasp_text = tokenize(left_row_text + " " + right_row_text)
        left_row_text = row_text[0:index1]
        right_row_text = row_text[index1+4:]
        left_tk_text = left_row_text.split()
        right_tk_text = right_row_text.split()
        crt_ctx_rmasp = []
        crt_asp = []
        crt_sent = []
        subposition = []
        left_subposition = []
        right_subposition = []
        left_tktext_idxes = []
        right_tktext_idxes = []
        local_idx2word = {}
        # the left part 2 ids.
        for w in left_tk_text:
            if w.lower() not in word2idx:
                if idx_cnt == pad_idx:
                    idx_cnt += 1
                word2idx[w.lower()] = idx_cnt
                idx_cnt += 1
            left_tktext_idxes.append(word2idx[w.lower()])
            local_idx2word[word2idx[w]] = w

        rasp = tmp[i + 1].lower().strip()
        crt_position = [index1, index1+len(rasp)]
        asps = rasp.split()
        for w in asps:
            if w.lower() not in word2idx:
                if idx_cnt == pad_idx:
                    idx_cnt += 1
                word2idx[w.lower()] = idx_cnt
                idx_cnt += 1
            crt_asp.append(word2idx[w.lower()])
            local_idx2word[word2idx[w]] = w
        # the right part 2 ids.
        for w in right_tk_text:
            if w.lower() not in word2idx:
                if idx_cnt == pad_idx:
                    idx_cnt += 1
                word2idx[w.lower()] = idx_cnt
                idx_cnt += 1
            right_tktext_idxes.append(word2idx[w.lower()])
            local_idx2word[word2idx[w]] = w

        # left + right 2 crt_ctx_rmasp
        crt_ctx_rmasp.extend(left_tktext_idxes)
        crt_ctx_rmasp.extend(right_tktext_idxes)

        left_subposition.append(len(crt_sent))
        crt_sent.extend(left_tktext_idxes)
        left_subposition.append(len(crt_sent))
        subposition.append(len(crt_sent))
        crt_sent.extend(crt_asp)
        right_subposition.append(len(crt_sent))
        subposition.append(len(crt_sent))
        crt_sent.extend(right_tktext_idxes)
        right_subposition.append(len(crt_sent))

        crt_lab = int(tmp[i + 2].strip())
        if crt_lab == -1:
            crt_lab = 1
        elif crt_lab == 1:
            crt_lab = 0
        elif crt_lab == 0:
            crt_lab = 2
        # text = row_text.lower().replace("$t$", tmp[i + 1]).split()
        # crt_ctx = []
        # for w in text:
        #     if w.lower() not in word2idx:
        #         if idx_cnt == pad_idx:
        #             idx_cnt += 1
        #         word2idx[w.lower()] = idx_cnt
        #         idx_cnt += 1
        #     crt_ctx.append(word2idx[w.lower()])

        if crt_lab != 3:
            if class_num == 2 and sample.label2pol(crt_lab) == 'neutral':
                continue
            sample.id = now_id
            sample.sent_id = now_id
            now_id += 1
            sample.aspect = rasp
            sample.text = row_text.replace("$t$", tmp[i + 1].strip())
            sample.aspect_idxes = crt_asp
            sample.text_idxes = crt_sent
            sample.left_context_idxes = left_tktext_idxes
            sample.right_context_idxes = right_tktext_idxes
            sample.context_idxes = crt_ctx_rmasp
            sample.label = crt_lab
            sample.aspect_charpos = crt_position
            sample.aspect_wordpos = subposition
            sample.left_wordpos = left_subposition
            sample.right_wordpos = right_subposition
            sample.local_idx2word = local_idx2word
            samples.append(sample)

        # crt_sent = []
        # subposition = []
        # crt_sent.extend(left_tktext_idxes)
        # subposition.append(len(crt_sent))
        # crt_sent.extend(crt_asp)
        # subposition.append(len(crt_sent) - 1)
        # crt_sent.extend(right_tktext_idxes)
        # # the full sentence. consists of the idxes.
        # crt_sent = []
        # subposition = []
        # crt_sent.extend(left_tktext_idxes)
        # subposition.append(len(crt_sent))
        # crt_sent.extend(crt_asp)
        # subposition.append(len(crt_sent) - 1)
        # crt_sent.extend(right_tktext_idxes)
        # crt_position = [len(left), len(left) + len(rasp)]


        # print crt_pol_text
        # crt_lab = -1
        # if crt_pol_text == "1":
        #     crt_lab = 1
        # elif crt_pol_text == "-1":
        #     crt_lab = 2
        # elif crt_pol_text == "0":
        #     crt_lab = 0
        # else:
        #     crt_lab = 3
        # if crt_lab != 3:
        #             contexts.append(crt_ctx_rmasp)
        #             aspects.append(crt_asp)
        #             positons.append(crt_position)
        #             labels.append(crt_lab)
        #             rowtexts.append(row_text)
        #             rowaspects.append(rasp)
        #             fullsents.append(crt_sent)
        #             subpositions.append(subposition)
    # retdata = [contexts, aspects, labels, positons,
    #            rowtexts, rowaspects, fullsents, subpositions]
    # print  labels
    samplepack.samples = samples
    samplepack.init_id2sample()
    return samplepack, idx_cnt


def tokenize(strs):
    """
    the tokenizer
    """
    tmp_text = nltk.word_tokenize(strs)
    text = []
    for w in tmp_text:
        if w == "'s" or w == "'t" or w == "n't" or w == "t's":
            text.append(w)
        else:
            w = w.replace('-', ' - ')
            w = w.replace('.', ' . ')
            w = w.replace("'", " ' ")
            w = w.replace("' t", "n't")
            w = w.replace("*", " * ")
            w = w.replace("555", " 555 ")
            w = w.replace("+", " + ")
            w = w.replace("/", " / ")
            # w = w.replace("pm", " pm ")
            # w = w.replace("am", " am ")
            ws = nltk.word_tokenize(w)
            for sw in ws:
                text.append(sw)
    return text

mid_dong_train_data = "dong_train.data"
mid_dong_test_data = "dong_test.data"

if __name__ == '__main__':
    trian, test, word2idx = load_data("../datas/data/train.raw",
                                      "../datas/data/test.raw")
    dataset = []

    tr_d = trian.samples
    te_d = test.samples
        #  print tr_d
    data = tr_d + te_d
        #  print data
    dataset += data
    for i in range(len(dataset)):
        dataset[i].id = i
    rand_idx = np.random.permutation(len(dataset))
    alenth = len(rand_idx) / 3
    count = 0
    datas = []
    for i in xrange(3):
        cdatas = []
        for j in xrange(alenth):
            cdatas.append(dataset[rand_idx[count]])
            count += 1
        datas.append(cdatas)
        # solve last data
    for i in rand_idx[count:]:
            datas[-1].append(dataset[i])

    tmp = copy.deepcopy(datas)
    for i in xrange(3):
        datas = copy.deepcopy(tmp)
        test_datas = datas.pop(i)
        trian_datas = []
        path = "../datas/3cross/cross" + str(i + 1) + "/"
        for x in datas:
            trian_datas += x
        print len(test_datas)
        print len(trian_datas)
        samplepack_train = Samplepack()
        samplepack_train.samples = trian_datas
        samplepack_train.init_id2sample()
        samplepack_test = Samplepack()
        samplepack_test.samples = test_datas
        samplepack_test.init_id2sample()
        dump_file(
            [samplepack_train, path + mid_dong_train_data],
            [samplepack_test, path + mid_dong_test_data]
        )

    # for i in xrange(3):
    #     datas = copy.deepcopy(tmp)
    #     train_datas = datas.pop(i)
    #     # train_datas2 = datas.pop(i)
    #     # train_datas = train_datas1+train_datas2
    #     test_datas = []
    #     path = "../datas/3train/cross" + str(i + 1) + "/"
    #     for x in datas:
    #         test_datas += x
    #     print len(test_datas)
    #     print len(train_datas)
    #     samplepack_train = Samplepack()
    #     samplepack_train.samples = train_datas
    #     samplepack_train.init_id2sample()
    #     samplepack_test = Samplepack()
    #     samplepack_test.samples = test_datas
    #     samplepack_test.init_id2sample()
    #     dump_file(
    #         [samplepack_train, path + mid_dong_train_data],
    #         [samplepack_test, path + mid_dong_test_data]
    #     )
