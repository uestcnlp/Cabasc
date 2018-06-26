def cau_acc(pred, labels):
    '''
    Calculate the accuracy. 
    pred.shape = [batch_size]
    pred: the predict labels. 

    labels.shape = [batch_size]
    labels: the gold labels. 
    '''
    acc = 0.0
    wrong_ids = []
    for i in xrange(len(labels)):
        if labels[i] == pred[i]:
            acc += 1.0
        else:
            wrong_ids.append(i)
    acc /= len(labels)
    return acc, wrong_ids

def cau_samples_acc(samples):
    acc = 0.0
    for sample in samples:
        if sample.is_pred_right():
            acc += 1.0
    acc /= len(samples)
    return acc
