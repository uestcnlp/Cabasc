import xml.etree.ElementTree as ET

import nltk
import copy
import numpy as np
from util.FileDumpLoad import dump_file
from data_prepare.entity.sample import Sample
from data_prepare.entity.samplepack import Samplepack
import random

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



def _load_data(file_path, word2idx, idx_cnt, pad_idx, class_num):
    tree = ET.parse(file_path)
    root = tree.getroot()

    samplepack = Samplepack()
    samples = []
    now_id = 0

    for sentence in root:
        row_text = sentence.find("text").text.lower()
        sent_id = sentence.get("id")
        for asp_terms in sentence.iter('aspectTerms'):
            # iter the aspects of on sentence.
            for asp_term in asp_terms.findall('aspectTerm'):
                sample = Sample()
                rasp = asp_term.get("term").lower()
                asps = tokenize(rasp)

                crt_pol_text = asp_term.get("polarity")
                crt_from = int(asp_term.get("from"))
                crt_to = int(asp_term.get("to"))
                crt_position = [crt_from, crt_to]

                # remove the aspect from the context. consists of the idxes.
                crt_ctx_rmasp = []
                crt_asp = []
                crt_sent = []
                subposition = []
                left_subposition = []
                right_subposition = []
                left_tktext_idxes = []
                right_tktext_idxes = []
                local_idx2word = {}

                left_row_text = row_text[0:crt_from]
                right_row_text = row_text[crt_to:]
                # rmasp_text = tokenize(left_row_text + " " + right_row_text)
                left_tk_text = tokenize(left_row_text)
                right_tk_text = tokenize(right_row_text)

                # the left part 2 ids.
                for w in left_tk_text:
                    if w not in word2idx:
                        if idx_cnt == pad_idx:
                            idx_cnt += 1
                        word2idx[w] = idx_cnt
                        idx_cnt += 1
                    left_tktext_idxes.append(word2idx[w])
                    local_idx2word[word2idx[w]] = w

                # the aspect.
                for w in asps:
                    if w not in word2idx:
                        if idx_cnt == pad_idx:
                            idx_cnt += 1
                        word2idx[w] = idx_cnt
                        idx_cnt += 1
                    crt_asp.append(word2idx[w])
                    local_idx2word[word2idx[w]] = w

                # the right part 2 ids.
                for w in right_tk_text:
                    if w not in word2idx:
                        if idx_cnt == pad_idx:
                            idx_cnt += 1
                        word2idx[w] = idx_cnt
                        idx_cnt += 1
                    right_tktext_idxes.append(word2idx[w])
                    local_idx2word[word2idx[w]] = w

                # left + right 2 crt_ctx_rmasp
                crt_ctx_rmasp.extend(left_tktext_idxes)
                crt_ctx_rmasp.extend(right_tktext_idxes)

                # the full sentence. consists of the idxes.
                left_subposition.append(len(crt_sent))
                crt_sent.extend(left_tktext_idxes)
                left_subposition.append(len(crt_sent))
                subposition.append(len(crt_sent))
                crt_sent.extend(crt_asp)
                right_subposition.append(len(crt_sent))
                subposition.append(len(crt_sent))
                crt_sent.extend(right_tktext_idxes)
                right_subposition.append(len(crt_sent))

                crt_lab = sample.pol2label(crt_pol_text)

                if crt_lab != 3:
                    if class_num == 2 and sample.label2pol(crt_lab) == 'neutral':
                        continue

                    sample.id = now_id
                    now_id += 1
                    sample.sent_id = sent_id
                    sample.aspect = rasp
                    sample.text = row_text
                    sample.aspect_idxes = crt_asp
                    sample.text_idxes =  crt_sent
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


# root_path = '/home/herb/code'
# project_name = '/Basic'
# res_train_path = root_path + project_name +  '/datas/data/Restaurants_Train.xml'
# res_test_path = root_path + project_name +  '/datas/data/Restaurants_Test_Gold.xml'
# lap_train_path = root_path + project_name +  '/datas/data/Laptops_Train.xml'
# lap_test_path = root_path + project_name +  '/datas/data/Laptops_Test_Gold.xml'
#
# if __name__ == '__main__':
#     train_data, test_data, word2idx = load_data(
#         lap_train_path,
#         lap_test_path,
#         class_num=2
#     )
#     print len(train_data.samples)
#     print len(test_data.samples)
mid_res_train_data = "res_train.data"
mid_res_test_data = "res_test.data"
mid_res_emb_dict = "res_emb_dict.data"
mid_res_word2idx = "res_mid_word2idx.data"
mid_lap_train_data = "lap_train.data"
mid_lap_test_data = "lap_test.data"
mid_lap_emb_dict = "lap_emb_dict.data"
mid_lap_word2idx = "lap_mid_word2idx.data"
mid_res_cat_emb_dict = "datas/res_cat_emb_dict.data"
mid_res_cat_word2idx = "datas/res_cat_mid_word2idx.data"
if __name__ == '__main__':
    trian, test, word2idx = load_data("../datas/data/Laptops_Train.xml",
                                      "../datas/data/Laptops_Test_Gold.xml")
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
    alenth = len(rand_idx) / 10
    count = 0
    datas = []
    for i in xrange(10):
        cdatas = []
        for j in xrange(alenth):
            cdatas.append(dataset[rand_idx[count]])
            count += 1
        datas.append(cdatas)
        # solve last data
    for i in rand_idx[count:]:
            datas[-1].append(dataset[i])

    tmp = copy.deepcopy(datas)
    # for i in xrange(10):
    #     datas = copy.deepcopy(tmp)
    #     test_datas = datas.pop(i)
    #     trian_datas = []
    #     path = "../datas/10train/cross" + str(i + 1) + "/"
    #     for x in datas:
    #         trian_datas += x
    #     print len(test_datas)
    #     print len(trian_datas)
    #     samplepack_train = Samplepack()
    #     samplepack_train.samples = trian_datas
    #     samplepack_train.init_id2sample()
    #     samplepack_test = Samplepack()
    #     samplepack_test.samples = test_datas
    #     samplepack_test.init_id2sample()
    #     dump_file(
    #         [samplepack_train, path + mid_res_train_data],
    #         [samplepack_test, path + mid_res_test_data]
    #     )
    # rand_idx = np.random.permutation(10)
    for i in xrange(10):
        datas = copy.deepcopy(tmp)
        train_datas = datas.pop(i)
        # train_datas2 = datas.pop(i)
        # train_datas = train_datas1+train_datas2
        test_datas = []
        path = "../datas/10train/cross" + str(i + 1) + "/"
        for x in datas:
            test_datas += x
        print len(test_datas)
        print len(train_datas)
        samplepack_train = Samplepack()
        samplepack_train.samples = train_datas
        samplepack_train.init_id2sample()
        samplepack_test = Samplepack()
        samplepack_test.samples = test_datas
        samplepack_test.init_id2sample()
        dump_file(
            [samplepack_train, path + mid_lap_train_data],
            [samplepack_test, path + mid_lap_test_data]
        )