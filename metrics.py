import math

import tensorflow as tf
from sklearn.metrics import matthews_corrcoef

def metric_acc(tp, fp, fn, tn):
    return (tp + tn) / (tp + fn + fp + tn)


def metric_f1(tp, fp, fn, tn):
    return (2 * tp) / ((2 * tp) + fp + fn)


def metric_precision(tp, fp):
    return tp / (tp + fp)


def metric_recall(tp, fn):
    return tp / (tp + fn)


def sum (v,col):
    s = 0
    for i in range (len(v)):
        s = s + v[i][col]
    return s

def sum_prod(table, col1, col2):
    s = 0
    for i in range (len(table)):
        s = s + (table[i][col1] * table[i][col2])
    return s
def metric_mcc(table):
    s = sum(table,0)
    c = sum(table, 2)
    tp = sum_prod(table, 0, 1)
    num = (s * c) - tp
    s2 = s * s
    p2 = sum_prod(table, 1, 1)
    t2 = sum_prod(table, 0,0)
    den = math.sqrt((s2-p2)*(s2-t2))
    mcc = num / den
    return mcc

def format_table (number_of_classes, correct_labels, predicted_labels):
    max = number_of_classes
    table = []
    for i in range(max):
        line = []
        line.append(0)
        line.append(0)
        line.append(0)
        table.append(line)
    for i in range(len(correct_labels)):
        (table[correct_labels[i]])[0] += 1
        (table[predicted_labels[i]])[1] += 1
        if correct_labels[i] == predicted_labels[i]:
            (table[correct_labels[i]])[2] += 1
    print(table)
    return table

def show_metrics(y_true, y_pred):
    NUMBER_OF_CLASSES = 4
    correct_labels = tf.concat(y_true, axis=0)
    predicted_labels = tf.concat(y_pred, axis=0)

    fp = tf.keras.metrics.FalsePositives()
    fp.update_state(correct_labels, predicted_labels)
    fn = tf.keras.metrics.FalseNegatives()
    fn.update_state(correct_labels, predicted_labels)
    tp = tf.keras.metrics.TruePositives()
    tp.update_state(correct_labels, predicted_labels)
    tn = tf.keras.metrics.TrueNegatives()
    tn.update_state(correct_labels, predicted_labels)

    overallAccuracy = tf.keras.metrics.Accuracy()
    overallAccuracy.update_state(correct_labels, predicted_labels)
    overallCategoricalAccuracy = tf.keras.metrics.CategoricalAccuracy()
    overallCategoricalAccuracy.update_state(correct_labels, predicted_labels)
    overallRecall = tf.keras.metrics.Recall()
    overallRecall.update_state(correct_labels, predicted_labels)
    overallPrecision = tf.keras.metrics.Precision()
    overallPrecision.update_state(correct_labels, predicted_labels)
    overallF1 = (2 * (overallPrecision.result() * overallRecall.result()) ) / (overallPrecision.result() + overallRecall.result())
    table = format_table(NUMBER_OF_CLASSES, correct_labels, predicted_labels)
    mcc = metric_mcc(table)
    print(correct_labels)
    print(predicted_labels)
    print('\nOverall metrics')
    print('Accuracy:', overallAccuracy.result())
    print('Categorical Accuracy:', overallCategoricalAccuracy.result())
    print('Recall:', overallRecall.result())
    print('F1 Score:', overallF1)
    print('Precision:', overallPrecision.result())
    print('MCC:', mcc)

    confusion_matrix = tf.math.confusion_matrix(correct_labels, predicted_labels)

    tps = []
    tns = []
    fps = []
    fns = []
    dim = len(confusion_matrix[0])


    for i in range(dim):
        tp = 0
        for j in range(dim):
            if i == j:
                tps.append(tf.keras.backend.get_value(confusion_matrix[i][j]))
                tnsc = 0
                for k in range(dim):
                    for l in range(dim):
                        if k != i and l != j:
                            tnsc += tf.keras.backend.get_value(confusion_matrix[k][l])
                tns.append(tnsc)
                fnsc = 0
                for k in range(dim):
                    if k != i:
                        fnsc += tf.keras.backend.get_value(confusion_matrix[i][k])
                fns.append(fnsc)
                fpsc = 0
                for k in range(dim):
                    if k != j:
                        fpsc += tf.keras.backend.get_value(confusion_matrix[k][j])
                fps.append(fpsc)
    print("\n\nConfusion Matrix")
    print(confusion_matrix)
    for i in range(dim):
        acc = metric_acc(tps[i], fps[i], fns[i], tns[i])
        f1 = metric_f1(tps[i], fps[i], fns[i], tns[i])
        precision = metric_precision(tps[i],fps[i])
        recall = metric_recall(tps[i],fns[i])

        print('\n\nClass ' + str(i))
        print('\nTrue Positives: ' + str(tps[i]))
        print('\nTrue Negatives: ' + str(tns[i]))
        print('\nFalse Positives: ' + str(fps[i]))
        print('\nFalse Negatives: ' + str(fns[i]))

        print('\nAccuracy: ' + str(acc))
        print('\nF1 Score: ' + str(f1))
        print('\nPrecision: ' + str(precision))
        print('\nRecall: ' + str(recall))
