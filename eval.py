from collections import defaultdict
import sys

RELATION_TYPES = ['OrgBased_In', 'Located_In', 'Live_In', 'Work_For', 'Kill']


def recall(gold_annotations, predicted_annotations, relation_types):
    correct = incorrect = 0.0
    not_predicted = []
    for sent_id, gold_sent_dict in gold_annotations.items():
        for rel_type, gold_rel_list in gold_sent_dict.items():
            if rel_type not in relation_types:
                continue
            for gold_rel in gold_rel_list:
                if comp_relation(gold_rel, predicted_annotations[sent_id][rel_type], True):
                    correct += 1
                else:
                    incorrect += 1
                    not_predicted.append((sent_id, gold_rel))
    with open('didnt make it to recall list', 'w') as file:
        for annot in not_predicted:
            file.write(str(annot) + '\n')
    return correct / (correct + incorrect)


def precision(gold_annotations, predicted_annotations, relation_types):
    in_accurate = []
    correct = incorrect = 0.0
    for sent_id, predicted_sent_dict in predicted_annotations.items():
        for rel_type, predicted_rel_list in predicted_sent_dict.items():
            if rel_type not in relation_types:
                continue
            for predicted_rel in predicted_rel_list:
                if comp_relation(predicted_rel, gold_annotations[sent_id][rel_type], False):
                    correct += 1
                else:
                    incorrect += 1
                    in_accurate.append((sent_id, predicted_rel))
    with open('didnt make it to precision list', 'w') as file:
        for annot in in_accurate:
            file.write(str(annot) + '\n')
    return correct / (correct + incorrect)


def F1(gold_annotations, predicted_annotations, relation_types):
    R = recall(gold_annotations, predicted_annotations, relation_types)
    P = precision(gold_annotations, predicted_annotations, relation_types)
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


def comp_relation(relation, relation_list, is_gold):
    if relation in relation_list:
        return True
    if is_gold:
        if [relation[0][:-1], relation[1]] in relation_list:
            return True
        if [relation[0], relation[1][:-1]] in relation_list:
            return True
        return False
    else:
        if [relation[0] + '.', relation[1]] in relation_list:
            return True
        if [relation[0], relation[1] + '.'] in relation_list:
            return True
        return False


def general_measure(gold_annotations, predicted_annotations):
    f1, r, p = F1(gold_annotations, predicted_annotations, RELATION_TYPES)
    print('General Precision: ' + str(p))
    print('General Recall: ' + str(r))
    print('General F1: ' + str(f1))
    print('\n')


def rel_type_measure(rel_type, gold_annotations, predicted_annotations):
    f1, r, p = F1(gold_annotations, predicted_annotations, [rel_type])
    print(rel_type + ' Precision: ' + str(p))
    print(rel_type + ' Recall: ' + str(r))
    print(rel_type + ' F1: ' + str(f1))


if __name__ == '__main__':
    gold_file, predicted_file = sys.argv[1], sys.argv[2]
    g_annotations, p_annotations = load_annotations(gold_file), load_annotations(predicted_file)
    general_measure(g_annotations, p_annotations)
    rel_type_measure('Live_In', g_annotations, p_annotations)