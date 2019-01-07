GOLD = 'Annotations'
TRAIN = '/train'
DEV = '/dev'


def load_gold(g_name):
    with open(g_name, 'r') as file:
        return {line.split()[0]: line.split()[1:3] for line in file}


def eval_pred_with_gold(preds, golds):
    good = bad = 0.0
    for pred, gold in zip(preds, golds):
        good += 1 if pred == gold else 0
        bad += 1 if pred != gold else 0
    print("precision: " + str(100 * (good / (good + bad))))