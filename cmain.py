# coding=utf-8
from optparse import OptionParser
import tensorflow as tf
from data_prepare.semeval14data_read import load_data
from data_prepare.semeval14data_read_dong import load_data as load_data_dong
from data_prepare.load_dict import load_glove
from data_prepare.load_dict import load_ssweu
from data_prepare.load_dict import load_random
from util.Config import read_conf
from util.FileDumpLoad import dump_file, load_file
from util.Randomer import Randomer

# the data path.
root_path = '/home/herb/code'
project_name = '/WWW18'
res_train_path = root_path + project_name + '/datas/data/Restaurants_Train.xml'
res_test_path = root_path + project_name + '/datas/data/Restaurants_Test_Gold.xml'
lap_train_path = root_path + project_name + '/datas/data/Laptops_Train.xml'
lap_test_path = root_path + project_name + '/datas/data/Laptops_Test_Gold.xml'
dong_train_path = root_path + project_name + '/datas/data/train.raw'
dong_test_path = root_path + project_name + '/datas/data/test.raw'
pre_train = "/home/herb/code/data/glove.42B.300d.txt"
#sswe_path = '/home/herb/code/data/glove.twitter.27B.50d.txt'
#sswe_path = '/home/herb/code/data/glove.6B.50d.txt'
sswe_path = '/home/herb/code/data/sswe-u.txt'

# the pretreatment data path.
mid_res_train_data = "res_train.data"
mid_res_test_data = "res_test.data"
mid_res_emb_dict = "res_emb_dict.data"
mid_res_sswe_dict = 'res_sswe_dict.data'
mid_res_rand_dict = "res_rand_dict.data"
mid_res_word2idx = "res_mid_word2idx.data"
mid_lap_train_data = "lap_train.data"
mid_lap_test_data = "lap_test.data"
mid_lap_emb_dict = "lap_emb_dict.data"
mid_lap_sswe_dict = "lap_sswe_dict.data"
mid_lap_rand_dict = "lap_rand_dict.data"
mid_lap_word2idx = "lap_mid_word2idx.data"
mid_dong_train_data = "dong_train.data"
mid_dong_test_data = "dong_test.data"
mid_dong_emb_dict = "dong_emb_dict.data"
mid_dong_sswe_dict = "dong_sswe_dict.data"
mid_dong_rand_dict = 'dong_rand_dict.data'
mid_dong_word2idx = "dong_mid_word2idx.data"
mid_res_cat_emb_dict = "datas/res_cat_emb_dict.data"
mid_res_cat_word2idx = "datas/res_cat_mid_word2idx.data"


def load_tt_datas(config={}, reload=True):
    '''
    loda data.
    config: 获得需要加载的数据类型，放入pre_embedding.
    nload: 是否重新解析原始数据
    '''
    if reload:
        print "reload the datasets."
        if config['dataset'] == 'lap':
            train_data, test_data, word2idx = load_data(
                lap_train_path,
                lap_test_path,
                class_num=config['class_num']
            )

        elif config['dataset'] == 'res':
            train_data, test_data, word2idx = load_data(
                res_train_path,
                res_test_path,
                class_num=config['class_num']
            )
        elif config['dataset'] == 'dong':
            train_data, test_data, word2idx = load_data_dong(
                dong_train_path,
                dong_test_path,
                class_num=config['class_num']
            )

        emb_dict = load_glove(
            pre_train,
            word2idx,
            init_std=config['emb_stddev']
        )

        sswe_dict = load_ssweu(
            sswe_path,
            word2idx,
            init_std=config['emb_stddev']
        )

        rand_dict = load_random(
            '',
            word2idx,
            init_std=config['emb_stddev']
        )
        # dump the pretreatment data.
        if config['class_num'] == 2:
            path = 'datas/mid_data_2classes/'
        else:
            path = 'datas/mid_data_3classes/'
        if config['dataset'] == 'lap':
            dump_file(
                [train_data, path + mid_lap_train_data],
                [test_data, path + mid_lap_test_data],
                [emb_dict, path + mid_lap_emb_dict],
                [word2idx, path + mid_lap_word2idx],
                [sswe_dict, path + mid_lap_sswe_dict],
                [rand_dict, path+ mid_lap_rand_dict]
            )
        elif config['dataset'] == 'res':
            dump_file(
                [train_data, path + mid_res_train_data],
                [test_data, path + mid_res_test_data],
                [emb_dict, path + mid_res_emb_dict],
                [word2idx, path + mid_res_word2idx],
                [sswe_dict, path + mid_res_sswe_dict],
                [rand_dict, path + mid_res_rand_dict]
            )
        elif config['dataset'] == 'dong':
            dump_file(
                [train_data, path + mid_dong_train_data],
                [test_data, path + mid_dong_test_data],
                [emb_dict, path + mid_dong_emb_dict],
                [word2idx, path + mid_dong_word2idx],
                [sswe_dict, path + mid_dong_sswe_dict],
                [rand_dict,path + mid_dong_rand_dict]
            )
    else:
        print "not reload the datasets."
        if config['class_num'] == 2:
            path = 'datas/mid_data_2classes/'
        else:
            path = 'datas/mid_data_3classes/'

        if config['dataset'] == 'lap':
            datas = load_file(
                path + mid_lap_train_data,
                path + mid_lap_test_data,
                path + mid_lap_emb_dict,
                path + mid_lap_sswe_dict,
                path + mid_lap_rand_dict
            )
        elif config['dataset'] == 'res':
            datas = load_file(
                path + mid_res_train_data,
                path + mid_res_test_data,
                path + mid_res_emb_dict,
                path + mid_res_sswe_dict,
                path + mid_res_rand_dict
            )
        elif config['dataset'] == 'dong':
            datas = load_file(
                path + mid_dong_train_data,
                path + mid_dong_test_data,
                path + mid_dong_emb_dict,
                path + mid_dong_sswe_dict,
                path + mid_dong_rand_dict
            )

        train_data = datas[0]
        test_data = datas[1]
        emb_dict = datas[2]
        sswe_dict = datas[3]
        rand_dict = datas[4]
    config['pre_embedding'] = emb_dict
    config['sswe_embedding'] = sswe_dict
    config['rand_embedding'] = rand_dict
    return train_data, test_data


def load_conf(model, modelconf):
    '''
    model: 需要加载的模型
    modelconf: model config文件所在的路径
    '''
    # load model config
    model_conf = read_conf(model, modelconf)
    if model_conf is None:
        raise Exception("wrong model config path.", model_conf)
    module = model_conf['module']
    obj = model_conf['object']
    params = model_conf['params']
    params = params.split("/")
    paramconf = ""
    model = params[-1]
    for line in params[:-1]:
        paramconf += line + "/"
    paramconf = paramconf[:-1]
    # load super params.
    param_conf = read_conf(model, paramconf)
    return module, obj, param_conf


def option_parse():
    '''
    parse the option.
    '''
    parser = OptionParser()
    parser.add_option(
        "-m",
        "--model",
        action='store',
        type='string',
        dest="model",
        default='memnn'
    )
    parser.add_option(
        "-d",
        "--dataset",
        action='store',
        type='string',
        dest="dataset",
        default='res'
    )
    parser.add_option(
        "-r",
        "--reload",
        action='store_true',
        dest="reload",
        default=False
    )
    parser.add_option(
        "-c",
        "--classnum",
        action='store',
        type='int',
        dest="classnum",
        default=3
    )
    parser.add_option(
        "-a",
        "--nottrain",
        action='store_true',
        dest="not_train",
        default=False
    )
    parser.add_option(
        "-n",
        "--notsavemodel",
        action='store_true',
        dest="not_save_model",
        default=False
    )
    parser.add_option(
        "-p",
        "--modelpath",
        action='store',
        type='string',
        dest="model_path",
        default='/home/herb/code/WWW18/ckpt/seq2seqlm.ckpt-3481-201709251759-lap'
    )
    parser.add_option(
        "-i",
        "--inputdata",
        action='store',
        type='string',
        dest="input_data",
        default='test'
    )
    (option, args) = parser.parse_args()
    return option


def main(options, modelconf="config/model.conf"):
    '''
    model: 需要加载的模型
    dataset: 需要加载的数据集
    reload: 是否需要重新加载数据，yes or no
    modelconf: model config文件所在的路径
    class_num: 分类的类别
    use_term: 是否是对aspect term 进行分类
    '''
    model = options.model
    dataset = options.dataset
    reload = options.reload
    class_num = options.classnum
    is_train = not options.not_train
    is_save = not options.not_save_model
    model_path = options.model_path
    input_data = options.input_data

    module, obj, config = load_conf(model, modelconf)
    config['model'] = model
    config['dataset'] = dataset
    config['class_num'] = class_num
    train_data, test_data = load_tt_datas(config, reload)
    module = __import__(module, fromlist=True)

    # setup randomer
    Randomer.set_stddev(config['stddev'])
    while(1):
        with tf.Graph().as_default():
            # build model
            model = getattr(module, obj)(config)
            model.build_model()
            if is_save or not is_train:
                saver = tf.train.Saver(max_to_keep=30)
            else:
                saver = None
            # run
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                if is_train:
                    print dataset
                    if dataset == "lap":
                        acc = model.train(sess, train_data, test_data, saver, threshold_acc=config['lap_threshold_acc'])
                        if acc > 0.7449:
                            break
                    elif dataset == "res":
                        model.train(sess, train_data, test_data, saver, threshold_acc=config['res_threshold_acc'])
                    elif dataset == "dong":
                        model.train(sess, train_data, test_data, saver, threshold_acc=config['dong_threshold_acc'])

                else:
                    if input_data is "test":
                        sent_data = test_data
                    elif input_data is "train":
                        sent_data = train_data
                    else:
                        sent_data = test_data
                    saver.restore(sess, model_path)
                    model.test(sess, sent_data)


if __name__ == '__main__':
    options = option_parse()
    main(options)
