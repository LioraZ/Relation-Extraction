import spacy
import itertools
import pickle
import sys
from sklearn.svm import SVC
from re_svm import SVM
import utils
import numpy as np


def get_dev_data(possible_relations):
    return [np.concatenate([ent1[0], ent2[0]]) for sent_ents in possible_relations.values() for rel_type, relation_type_list
            in sent_ents.items() for (ent1, ent2) in relation_type_list]


def predict_dev(svm, processed_data):
    ner_vecs = {sent_id: utils.get_entity_vecs(nlp, sentence) for sent_id, sentence in processed_data.items()}
    possible_relations, word_format = utils.build_relation_data(ner_vecs)
    dev_set = get_dev_data(possible_relations)
    predictions = svm.predict(dev_set)
    return predictions, word_format


def has_relation(annotation, processed_data):
    sent_id, ent1, rel_type, ent2 = annotation
    if rel_type != 'Live_In':
        return False
    sent_tokens = processed_data[sent_id][utils.SENTENCE]
    print(sent_tokens)
    return False


def write_predicted_relations(f_name, predictions, entities_dict, processed_data):
    flattened_data = [[sent_id, ent1, rel_type, ent2] for sent_id, relations_dict in entities_dict.items()
                      for rel_type, rel_type_list in relations_dict.items() for (ent1, ent2) in rel_type_list]
    with open(f_name, 'w') as file:
        for annotation, prediction in zip(flattened_data, predictions):
            if prediction == 1 or (prediction == 0 and has_relation(annotation, processed_data)):
                file.write('\t'.join(annotation) + '\n')


if __name__ == '__main__':
    input_file, output_file = sys.argv[1], sys.argv[2]
    model = pickle.load(open('model_file', 'rb'))
    pr_data = utils.get_processed_data(input_file)
    nlp = spacy.load('en')
    dev_predictions, sent_dict = predict_dev(model, pr_data)
    write_predicted_relations(output_file, dev_predictions, sent_dict, pr_data)