from collections import defaultdict
import sys


def recall(gold_annotations, predicted_annotations):
    correct = incorrect = 0.0
    for sent_id, gold_sent_dict in gold_annotations.items():
        for rel_type, gold_rel_list in gold_sent_dict.items():
            for gold_rel in gold_rel_list:
                if gold_rel in predicted_annotations[sent_id][rel_type]:
                    correct += 1
                else:
                    incorrect += 1
    return correct / (correct + incorrect)


def precision(gold_annotations, predicted_annotations):
    correct = incorrect = 0.0
    for sent_id, predicted_sent_dict in predicted_annotations.items():
        for rel_type, predicted_rel_list in predicted_sent_dict.items():
            for predicted_rel in predicted_rel_list:
                if predicted_rel in gold_annotations[sent_id][rel_type]:
                    correct += 1
                else:
                    incorrect += 1
    return correct / (correct + incorrect)


def F1(gold_annotations, predicted_annotations):
    R = recall(gold_annotations, predicted_annotations)
    P = precision(gold_annotations, predicted_annotations)
    return 2 * (P * R) / (P + R), R, P


def load_annotations(f_name):
    relations_dict = defaultdict(lambda: defaultdict(list))
    relation_types = []
    with open(f_name, 'r') as file:
        for line in file:
            fields = line.strip('\n').split('\t')
            relation_types.append(fields[2])
            relations_dict[fields[0]][fields[2]] += [[fields[1], fields[3]]]
    return relations_dict


if __name__ == '__main__':
    gold_file, predicted_file = sys.argv[1], sys.argv[2]
    g_annotations, p_annotations = load_annotations(gold_file), load_annotations(predicted_file)
    f1, r, p = F1(g_annotations, p_annotations)
    print('Precision: ' + str(p))
    print('Recall: ' + str(r))
    print('F1: ' + str(f1))