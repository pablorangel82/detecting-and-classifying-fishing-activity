import math

# Also known as Specificity
def true_negative_rate(tn, fp):
    if (tn + fp) == 0:
        return 0
    return tn / (tn + fp)


def metric_acc(tp, fp, fn, tn):
    if (tp + fn + fp + tn) == 0:
        return 0
    return (tp + tn) / (tp + fn + fp + tn)


def metric_f1(precision, recall):
    if (precision + recall) == 0:
        return 0
    return 2 * ((precision * recall) / (precision + recall))


def metric_precision(tp, fp):
    if (tp + fp) == 0:
        return 0
    return tp / (tp + fp)


# Also known as True Positive Rate or Sensitivity or Recall
def metric_true_positive_rate(tp, fn):
    if (tp + fn) == 0:
        return 0
    return tp / (tp + fn)


def metric_false_positive_rate(fp, tn):
    if (fp + tn) == 0:
        return 0
    return fp / (fp + tn)


def sum_vector(v):
    s = 0
    for i in range(len(v)):
        s = s + v[i]
    return s


def sum(v, col):
    s = 0
    for i in range(len(v)):
        s = s + v[i][col]
    return s


def sum_prod(table, col1, col2):
    s = 0
    for i in range(len(table)):
        s = s + (table[i][col1] * table[i][col2])
    return s


def metric_mcc(table):
    s = sum(table, 0)
    c = sum(table, 2)
    tp = sum_prod(table, 0, 1)
    num = (s * c) - tp
    s2 = s * s
    p2 = sum_prod(table, 1, 1)
    t2 = sum_prod(table, 0, 0)
    w = (s2 - p2)
    z = (s2 - t2)
    den = math.sqrt(w) * math.sqrt(z)
    if den == 0:
        return -1
    mcc = num / den
    return mcc


def format_table(number_of_classes, correct_labels, predicted_labels):
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

def comparision(obtained, fishnet, gfw):
    obtained = obtained * 100
    eval = 'Worse of all'
    if obtained > fishnet and obtained > gfw:
        eval = 'Best of all'
    else:
        if obtained > fishnet:
            eval = 'Better than fishnet only'
        if obtained > gfw:
            eval = 'Better than gfw only'
    return eval

def show_metrics(confusion_matrix, table):
    classes = ['Drifting longlines', 'Fixed Gear', 'Not fishing', 'Purse seines', 'Trawlers']
    tps = []
    tns = []
    fps = []
    fns = []
    dim = len(confusion_matrix[0])
    print('\nSummary Table')
    print(table)
    for i in range(dim):
        tp = 0
        for j in range(dim):
            if i == j:
                tps.append(confusion_matrix[i][j])
                tnsc = 0
                for k in range(dim):
                    for l in range(dim):
                        if k != i and l != j:
                            tnsc += confusion_matrix[k][l]
                tns.append(tnsc)
                fnsc = 0
                for k in range(dim):
                    if k != i:
                        fnsc += confusion_matrix[i][k]
                fns.append(fnsc)
                fpsc = 0
                for k in range(dim):
                    if k != j:
                        fpsc += confusion_matrix[k][j]
                fps.append(fpsc)
    print("\nConfusion Matrix")
    print(confusion_matrix)
    fishnet = []
    fishnet.append([92.63,92.23,91.78,92])
    fishnet.append([97.62,97.35,97.36,97.35])
    fishnet.append([0,0,0,0])
    fishnet.append([97.58,96.21,95.34,95.77])
    fishnet.append([98.27,96.13,95.18,95.65])

    gfw = []
    gfw.append([92,94,91,93])
    gfw.append([95,88,97,90])
    gfw.append([0,0,0,0])
    gfw.append([78,81,95,79])
    gfw.append([98,94,96,96])

    for i in range(dim):
        acc = metric_acc(tps[i], fps[i], fns[i], tns[i])
        precision = metric_precision(tps[i], fps[i])
        recall = metric_true_positive_rate(tps[i], fns[i])
        f1 = metric_f1(precision, recall)

        acc_eval = comparision(acc, fishnet[i][0], gfw[i][0])
        prec_eval = comparision(precision, fishnet[i][1], gfw[i][1])
        recall_eval = comparision(recall, fishnet[i][2], gfw[i][2])
        f1_eval = comparision(f1, fishnet[i][3], gfw[i][3])

        print('\nClass ' + classes[i])
        print('True Positives: ' + str(tps[i]))
        print('True Negatives: ' + str(tns[i]))
        print('False Positives: ' + str(fps[i]))
        print('False Negatives: ' + str(fns[i]))
        print('Metric | Value Obtained | FishNet | GFW')
        print('Accuracy: ' + str(acc*100) + ' | ' + str(fishnet[i][0]) + ' | ' + str(gfw[i][0]) + ' ' + acc_eval)
        print('Precision: ' + str(precision*100) + ' | ' + str(fishnet[i][1]) + ' | ' + str(gfw[i][1]) + ' ' + prec_eval)
        print('Recall: ' + str(recall*100) + ' | ' + str(fishnet[i][2]) + ' | ' + str(gfw[i][2]) + ' ' + recall_eval)
        print('F1 Score: ' + str(f1*100) + ' | ' + str(fishnet[i][3]) + ' | ' + str(gfw[i][3]) + ' ' + f1_eval)

    total_tp = sum_vector(tps)
    total_fp = sum_vector(fps)
    total_fn = sum_vector(fns)
    total_tn = sum_vector(tns)

    print('\nOverall metrics')
    print('Accuracy: ', metric_acc(total_tp, total_fp, total_fn, total_tn))
    recall = metric_true_positive_rate(total_tp, total_fn)
    precision = metric_precision(total_tp, total_fp)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', metric_f1(precision, recall))
    print('MCC:', metric_mcc(table))
