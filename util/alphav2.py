# coding=utf-8
import numpy as np
import os
randn = np.random.randn
from matplotlib import pyplot as plt
from pandas import *
def drawtable(word_list,attention_list,aspect,sentence,label,wrong_label,picname,savepath):
    idx = Index(word_list)
    nhop=len(attention[0])
    col=[]
    for x in xrange(nhop):
        col.append("hop"+str(x+1))
    df = DataFrame(np.array(attention_list), index=idx, columns=col)
    vals = np.around(df.values, 4)
    normal = plt.Normalize(vals.min(), vals.max()+0.2)
    nrows, ncols = len(idx) +1, len(df.columns)+1
    hcell, wcell = 1, 3
    hpad, wpad = 2, 5
    lenth=ncols*wcell+wpad
    width=nrows*hcell+hpad
    if lenth>width:
        fig = plt.figure(figsize=(lenth+wpad, lenth+wpad))
    else:
        fig = plt.figure(figsize=(width+wpad, width+wpad))
    ax = fig.add_subplot(1,1,1)

    ax.axis('off')
    the_table = ax.table(cellText=vals, rowLabels=df.index, colLabels=df.columns, loc='center',
                          cellColours=plt.cm.gray_r(normal(vals)))
    table_props = the_table.properties()
    table_cells = table_props['child_artists']
    for cell in table_cells:
        cell.set_fontsize(22)
        cell.set_height(0.03)
    s_sentence=sentence.split()
    s=""
    for i in xrange(len(s_sentence)):
        if (i+1)%16 == 0:
            s+=s_sentence[i]+"\n"
        else:
            s+=s_sentence[i]+" "
    ax.text(0, 0, str(s.replace('$','')),fontsize=30)
    if wrong_label== None :
        ax.set_title('aspect: ' + str(aspect) + '  ;label:' + str(right_label),fontsize=30)
    else:
        ax.set_title('aspect: ' + str(aspect) + '  ;label:' + str(right_label)+'  ;prediction:'+str(wrong_label),fontsize=30)
    if os.path.exists(savepath) != True:
        os.makedirs(savepath)
    fig.savefig(picname+".jpg")
    plt.close(fig)
def eachFile(filepath,fliter=None):
    pathDir =  os.listdir(filepath)
    file_path=[]
    for allDir in pathDir:
        if fliter != None:
            # 根据要画的文件名决定
            if fliter in allDir:
                child = allDir
                file_path.append(child)
        else:
            child = allDir
            file_path.append(child)
    return file_path
# path="attention/"
# FWNN_lap_rigt=path+"FwNN-lapalpha.right"
# FWNN_lap_wrong=path+"FwNN-lapalpha.wrong"
# FWNN_res_rigt=path+"FwNN-resalpha.right"
# FWNN_res_wrong=path+"FwNN-resalpha.wrong"
# MNN_res_rigt=path+"MemNN-resalpha.right"
# MNN_res_wrong=path+"MemNN-resalpha.wrong"
# MNN_lap_rigt=path+"MemNN-lapalpha.right"
# MNN_lap_wrong=path+"MemNN-lapalpha.wrong"

root_path='/home/herb/code/Seq2Seq/output/attention/'
root_pic='/home/herb/code/Seq2Seq/output/pic/'
fliter='55'
file_path=eachFile(root_path,fliter)

for path in file_path:

    file_tmp=[]
    words=[]
    id_list=[]
    aspect_list=[]
    sentence_list=[]
    right_label_l=[]
    wrong_label_l=[]
    print root_path+path
    file_object = open(root_path+path)
    
    id=0
    for line in file_object:
            #处理句子
            if len(line.strip().split(': '))>4:
                if words != []:
                    file_tmp.append(words)
                if id==5:
                    break
                id+=1
                processed = line.strip().split('\t')
                id_list.append(processed[0].split('id: ')[-1])
                sentence_list.append(processed[1].split('sentence: ')[-1])
                aspect_list.append(processed[2].split(': ')[-1])
                right_label_l.append(processed[3].split(': ')[-1])
                if len(line.strip().split('\t')) == 5:
                    wrong_label_l.append(processed[4].split(': ')[-1])
                else:
                    wrong_label_l.append(None)
                words=[]
            # 处理attention
            elif len(line.strip().replace(':','').split('\t'))>1:
                words.append(line)
            else:
                pass
    file_object.close( )
    savepath=root_pic+path.split('-')[0]+'/'+path.split('-')[1]+'/'+path.split('.')[-1]
    for word_seq,id,aspect,sentence,right_label,wrong_label in zip(file_tmp,id_list,aspect_list,sentence_list,right_label_l,wrong_label_l):
        attention=[]
        word_list=[]
        for word in word_seq:
            word=word.strip().replace(':','').split('\t')
            word_list.append(word.pop(0))
            attention.append([float(x) for x in word])
        picname=root_pic+path.split('-')[0]+'/'+path.split('-')[1]+'/'+path.split('.')[-1]+'/'+path.split('.')[0]+'-'+path.split('.')[-1]+"-"+str(id)
        print picname
        drawtable(word_list,attention,aspect,sentence,right_label,wrong_label,picname,savepath)

