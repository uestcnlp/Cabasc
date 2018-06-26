# coding=utf-8
import time
import cPickle as cp
import numpy as np
import sys, os
reload(sys)
sys.setdefaultencoding('utf-8')

def TIPrint(samples, config, is_pred_right = False, acc = {},print_att = False,Time = None):
    base_path = os.path.realpath(__file__)
    bps = base_path.split("/")[1:-2]
    base_path = "/"
    for bp in bps:
        base_path += bp + "/"
    base_path += 'output/'
    if Time is None:
        suf = time.strftime("%Y%m%d%H%M", time.localtime())
    else:
        suf = Time

    path = base_path + "text/" + config['model'] + "-" + config['dataset'] + "-" + suf
    if is_pred_right:
        path += ".right"
    else:
        path += ".wrong"
    print_txt(path, samples, config, is_pred_right, acc, print_att)
    return suf

def print_txt(path, samples, config, is_pred_right = False, acc = {}, print_att = False):
    '''
    写入文本数据，使用writer
    :param samples: 样本
    :param config: 模型参数
    :param acc: acc = {'max_acc':0.0, 'max_train_acc': 0.0}
    :return: None
    '''
    outfile = open(path, 'w')
    outfile.write('accuracy:\n')
    for k,v in acc.items():
        outfile.write(str(k) + ' :\t' + str(v) + '\n')

    outfile.write("\nconfig:\n")
    for k,v in config.items():
        outfile.write(str(k) + ' :\t' + str(v) + '\n')

    outfile.write("\nsample:\n")
    avg_output_dic = {"0": [], "1": [], "2": []}
    before_output_dic = {"0": [], "1": [], "2": []}
    after_output_dic = {"0": [], "1": [], "2": []}
    noadd_output_dic = {"0": [], "1": [], "2": []}
    output_dic = {"0": [], "1": [], "2": []}
    output_dic_p = {"0": [], "1": [], "2": []}
    for sample in samples:
        if sample.is_bestpred_right() == is_pred_right:
            outfile.write("id      :\t" + str(sample.id) + '\n')
            outfile.write("text    :\t" + str(sample.text) + '\n')
            outfile.write("aspect  :\t" + str(sample.aspect) + '\n')
            outfile.write("label   :\t" + str(sample.label) + '\t' + str(sample.label2pol(sample.label)) + '\n')
            outfile.write("predict :\t" + str(sample.best_pred) + '\t' + str(sample.label2pol(sample.best_pred)) + '\n')
            if print_att:
                for ext_key in sample.ext_matrix:
                    matrix = sample.ext_matrix[ext_key]
                    outfile.write("attention :\t" + str(ext_key) + '\n')
                    if ext_key == "left":
                        for i in range(len(sample.left_context_idxes)):
                            outfile.write(sample.local_idx2word[sample.left_context_idxes[i]] + " :\t")
                            for att in matrix:
                                outfile.write(str(att[i]) + " ")
                            outfile.write("\n")
                    elif ext_key == "right":
                        for i in range(len(sample.right_context_idxes)):
                            outfile.write(sample.local_idx2word[sample.right_context_idxes[i]] + " :\t")
                            for att in matrix:
                                outfile.write(str(att[i]) + " ")
                            outfile.write("\n")
                    elif ext_key == "out":
                        # outfile.write("output :\t" + str(matrix)  + '\n')
                        output_dic[str(sample.label)].append(matrix)
                        output_dic_p[str(sample.best_pred)].append(matrix)
                    elif ext_key == "before_add":
                        # outfile.write("output :\t" + str(matrix)  + '\n')
                        before_output_dic[str(sample.label)].append(matrix)
                    elif ext_key == "after_add":
                        # outfile.write("output :\t" + str(matrix)  + '\n')
                        after_output_dic[str(sample.label)].append(matrix)
                    elif ext_key == "avg_out":
                        # outfile.write("output :\t" + str(matrix)  + '\n')
                        avg_output_dic[str(sample.label)].append(matrix)
                    elif ext_key == "noadd_out":
                        # outfile.write("output :\t" + str(matrix)  + '\n')
                        noadd_output_dic[str(sample.label)].append(matrix)
                    else:
                        for i in range(len(sample.text_idxes)):
                            outfile.write(sample.local_idx2word[sample.text_idxes[i]]+" :\t")
                            for att in matrix:
                                outfile.write(str(att[i])+" ")
                            outfile.write("\n")
            outfile.write("\n")
    if len(output_dic["0"]) > 0:
        b_path = path.split('.')[0]
        suff =  path.split('.')[-1]
        b_path = b_path.split('/')[-1] + "-" + suff
        base_path = os.path.realpath(__file__)
        bps = base_path.split("/")[1:-2]
        base_path = "/"
        for bp in bps:
            base_path += bp + "/"
        base_path += 'output/binary/'
        b_path1 = base_path + b_path + "-out.data"
        print_binary(b_path1,datas=output_dic)
        b_path2 = base_path + b_path + "-avg.data"
        print_binary(b_path2, datas=avg_output_dic)
        b_path3 = base_path + b_path + "-before.data"
        print_binary(b_path3, datas=before_output_dic)
        b_path4 = base_path + b_path + "-after.data"
        print_binary(b_path4, datas=after_output_dic)
        b_path5 = base_path + b_path + "-noadd.data"
        print_binary(b_path5, datas=noadd_output_dic)
        b_path6 = base_path + b_path + "-pout.data"
        print_binary(b_path6, datas=output_dic_p)
    outfile.close()

def print_binary(path, datas):
    '''
    写入序列数据，用cPickle
    :param ids: 样本的id，需要写入文件系统的数据的id
    :param datas: datas = {'':[[]], ...}, [[]] 的第0个维度给出id. 需要根据ids从中挑选需要写入的数据重新构建字典
    :return: None
    '''
    dfile = file(path, 'w')
    cp.dump(datas, dfile)
    dfile.close()
    pass

