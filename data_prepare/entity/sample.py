#coding=utf-8

class Sample(object):
    '''
    一个样本。
    '''
    def __init__(self):
        self.id = -1
        self.sent_id = -1
        self.text = ''
        self.aspect = ''
        self.label = -1
        self.text_idxes = []
        self.aspect_idxes = []
        self.left_context_idxes = []
        self.right_context_idxes = []
        self.context_idxes = []
        self.aspect_charpos = [] # 方面在句子中字符级别的位置
        self.aspect_wordpos = [] # 方面在句子中词级别的位置. [start, end+1]
        self.local_idx2word = {} # global word index to word. Just the words in this sentence is storage.
        self.pred = -1  #当前模型预测结果
        self.best_pred = -1 #到目前位置最好的预测结果
        self.memory = []
        self.ext_matrix = {} # 额外数据，key是名字，value是矩阵。例如attention.
        self.pol2labels = {
            'neutral':2,
            'positive':0,
            'negative':1
        }
        self.label2pols = {
            2:'neutral',
            0:'positive',
            1:'negative'
        }

    def pol2label(self, pol):
        return self.pol2labels.get(pol, 3)

    def label2pol(self, label):
        return self.label2pols.get(label, 'unknow')

    def is_pred_right(self):
        return self.label == self.pred

    def is_bestpred_right(self):
        return self.label == self.best_pred

    def get_aspect_memory(self):
        if self.memory is None or len(self.memory) == 0:
            return []
        return self.memory[self.aspect_wordpos[0]:self.aspect_wordpos[1]]

    def __str__(self):
        ret = 'id: ' + str(self.id) + '\n'
        ret += 'text: ' + str(self.text) + '\n'
        ret += 'aspect: ' + str(self.aspect) + '\n'
        ret += 'label: ' + str(self.label2pol(self.label)) + '\n'
        ret += 'label: ' + str(self.label) + '\n'
        ret += 'text_idxes: ' + str(self.text_idxes) + '\n'
        ret += 'aspect_idxes: ' + str(self.aspect_idxes) + '\n'
        ret += 'context_idxes: ' + str(self.context_idxes) + '\n'
        ret += 'aspect_charpos: ' + str(self.aspect_charpos) + '\n'
        ret += 'aspect_wordpos: ' + str(self.aspect_wordpos) + '\n'
        return ret