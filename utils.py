import spacy
# from try_tomer import SVM
from sklearn.svm import SVC
import numpy as np
from eval import load_annotations
from collections import defaultdict

SENTENCE = 'sentence'
ENTITIES = 'ents'

RELATION_TYPES = {'OrgBased_In': [['ORG'], ['GPE']],
                  'Located_In': [['GPE'], ['GPE']],
                  'Live_In': [['PERSON'], ['GPE']],
                  'Work_For': [['PERSON'], ['ORG']],
                  'Kill': [['PERSON'], ['PERSON']]}
NER_TYPES = ['ORG', 'PERSON', 'GPE']


def get_processed_data(f_name):
    data = {}
    with open(f_name, 'r') as file:
        id = None
        for line in file:
            if line == '\n':
                continue
            fields = line.split()
            if fields[0] == '#id:':
                id = fields[1]
                data[id] = {}
                data[id][SENTENCE] = []
                data[id][ENTITIES] = {}
                continue
            data[id][SENTENCE] += [fields[1]]
            if fields[0] == '#text:':
                data[id][SENTENCE] = fields[1:]
                continue
            if fields[-1] != 'O':
                data[id][ENTITIES][int(fields[0]) - 1] = (fields[-1], fields[-2], fields[1])
        return data


def get_gold_relations(f_name):
    gold_annotations = load_annotations(f_name)
    return gold_annotations


def get_entity_vecs(nlp, sentence):
    sent_vecs = []
    entities = []
    ners = []
    parsed = nlp(' '.join(sentence[SENTENCE])).tensor
    curr_sentence = sentence[ENTITIES]
    for ent_id, (ent, iob, word) in curr_sentence.items():
        if iob == 'B':
            ners.append(ent)
            ent_vec = parsed[ent_id]
            sent_vecs.append(ent_vec)
            entities.append(word)
        if iob != 'B':
            sent_vecs[-1] += parsed[ent_id]
            entities[-1] += ' ' + word
    return sent_vecs, ners, entities


def build_relation_data(ner_vecs):
    possible_relations = {}
    word_relations = {}
    for sent_id, ner_list in ner_vecs.items():
        ner_dict = defaultdict(list)
        for vec, ner_tag, entity in zip(ner_list[0], ner_list[1], ner_list[2]):
            ner_dict[ner_tag] += [(vec, entity)]
        sent_relations = {}
        word_format = {}
        for rel_type, ner_tags in RELATION_TYPES.items():
            possible_ent1, possible_ent2 = ner_tags
            if any(ent in ner_dict for ent in possible_ent1) and any(ent in ner_dict for ent in possible_ent2):
                sent_relations[rel_type] = [(vec1[0], vec2[0]) for ent1 in possible_ent1 for vec1 in ner_dict[ent1] for ent2
                                                in possible_ent2 for vec2 in ner_dict[ent2]]
                word_format[rel_type] = [(vec1[1], vec2[1]) for ent1 in possible_ent1 for vec1 in ner_dict[ent1] for ent2
                                                in possible_ent2 for vec2 in ner_dict[ent2]]
        possible_relations[sent_id] = sent_relations
        word_relations[sent_id] = word_format
    return possible_relations, word_relations


def tag_possible_relations(gold_relations, possible_relations):
    train_set = []
    train_tags = []
    for sent_id, relations_dict in possible_relations.items():
        for rel_type, entity_list in relations_dict.items():
            for entities in entity_list:
                train_set.append(np.concatenate([entities[0][0], entities[1][0]]))
                if any([entities[0][1], entities[1][1]] == relation for relation in gold_relations[sent_id][rel_type]):
                    train_tags.append(1)
                else:
                    train_tags.append(0)
    return train_set, train_tags


"""def train_svm(processed_data):
    ner_vecs = {sent_id: get_entity_vecs(sentence) for sent_id, sentence in processed_data.items()}
    gold_relations = get_gold_relations('Annotations/train.txt')
    possible_relations = build_relation_data(ner_vecs)
    # annotations = get_relations_per_sen('Annotations/train.txt')
    train_set, train_tags = tag_possible_relations(gold_relations, possible_relations)
    svm = SVC()
    svm.fit(train_set, train_tags)
    return svm"""

