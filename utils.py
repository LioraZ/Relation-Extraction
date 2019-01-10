from nltk.tag import StanfordPOSTagger
import spacy
import nltk
import numpy as np
import random
import math
from eval import load_annotations
from collections import defaultdict

SENTENCE = 'sentence'
ENTITIES = 'entities'

"""RELATION_TYPES = {'OrgBased_In': [['ORGANIZATION'], ['GPE', 'FACILITY', 'LOCATION']],
                  'Located_In': [['GPE', 'FACILITY', 'LOCATION'], ['GPE', 'FACILITY', 'LOCATION']],
                  'Live_In': [['PERSON'], ['GPE', 'FACILITY', 'LOCATION']],
                  'Work_For': [['PERSON'], ['ORGANIZATION']],
                  'Kill': [['PERSON'], ['PERSON']]}"""

RELATION_TYPES = {'OrgBased_In': [['ORG'], ['GPE', 'FACILITY', 'LOC']],
                  'Located_In': [['GPE', 'FACILITY', 'LOC'], ['GPE', 'FACILITY', 'LOC']],
                  'Live_In': [['PERSON'], ['GPE', 'FACILITY', 'LOC']],
                  'Work_For': [['PERSON'], ['ORG']],
                  'Kill': [['PERSON'], ['PERSON']]}

# NER_TYPES = ['ORGANIZATION', 'PERSON', 'GPE', 'FACILITY', 'LOCATION', 'GSP', ]
NER_TYPES = ['ORG', 'PERSON', 'GPE', 'FACILITY', 'LOC']


def new_load_data(f_name):
    data = {}
    import os
    java_path = "C:/Program Files/Java/jdk1.8.0_121/bin/java.exe"
    os.environ['JAVAHOME'] = java_path
    st = StanfordPOSTagger('english-bidirectional-distsim.tagger', 'stanford-postagger.jar', encoding='utf-8')

    with open(f_name, 'r') as file:
        for line in file:
            fields = line.split('\t')
            sent_id = fields[0]
            """if sent_id == 'sent1656':
                print('yay')"""
            data[sent_id] = {}
            data[sent_id][SENTENCE] = fields[1].strip('\n').split()
            data[sent_id][ENTITIES] = {}
            tokenized_sent = nltk.sent_tokenize(fields[1])
            for sent in tokenized_sent:
                chunk_id = 0
                for chunk in nltk.ne_chunk(st.tag(nltk.word_tokenize(sent))):

                    if hasattr(chunk, 'label'):
                        data[sent_id][ENTITIES][chunk_id] = (chunk.label(), ' '.join(c[0] for c in chunk))
                        chunk_id += len([c[0] for c in chunk])
                        print(chunk.label(), ' '.join(c[0] for c in chunk))
                    else:
                        chunk_id += 1

                    # assert chunk_id < len(fields[1].split())
            #sent = st.tag(fields[1].split())
            #print(sent)
    return data


def get_new_entity_vecs(nlp, sentence):
    sent_vecs = []
    entities = []
    ent_ids = []
    ners = []
    parsed = nlp(' '.join(sentence[SENTENCE])).tensor
    curr_sentence = sentence[ENTITIES]
    for ent_id, (ent, word) in curr_sentence.items():
        if ent_id > 0 and sentence[SENTENCE][ent_id - 1] in ['Mrs.', 'Mr.', 'Ms.']:
            word = sentence[SENTENCE][ent_id - 1] + ' ' + word
        ners.append(ent)
        ent_vec = parsed[ent_id]
        sent_vecs.append(ent_vec)
        entities.append(word)
        ent_ids.append(ent_id)
        """if iob == 'B':
            if ent_id > 0 and sentence[SENTENCE][ent_id - 1] in ['Mrs.', 'Mr.', 'Ms.']:
                word = sentence[SENTENCE][ent_id - 1] + ' ' + word
            ners.append(ent)
            ent_vec = parsed[ent_id]
            sent_vecs.append(ent_vec)
            entities.append(word)
            ent_ids.append(ent_id)
        if iob != 'B':
            sent_vecs[-1] += parsed[ent_id]
            entities[-1] += ' ' + word"""
    return sent_vecs, ners, entities, ent_ids


def get_processed_data(f_name):
    # return new_load_data(f_name)
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
            if fields[0] == '#text:':
                # data[id][SENTENCE] = fields[1:]
                continue
            data[id][SENTENCE] += [fields[1]]
            if fields[-1] != 'O':
                data[id][ENTITIES][int(fields[0]) - 1] = (fields[-1], fields[-2], fields[1])
        return data


def get_gold_relations(f_name):
    gold_annotations = load_annotations(f_name)
    return gold_annotations


def get_entity_vecs(nlp, sentence):
    # return get_new_entity_vecs(nlp, sentence)
    sent_vecs = []
    entities = []
    ent_ids = []
    ners = []
    parsed = nlp(' '.join(sentence[SENTENCE])).tensor
    curr_sentence = sentence[ENTITIES]
    for ent_id, (ent, iob, word) in curr_sentence.items():
        if iob == 'B':
            if ent_id > 0 and sentence[SENTENCE][ent_id - 1] in ['Mrs.', 'Mr.', 'Ms.']:
                word = sentence[SENTENCE][ent_id - 1] + ' ' + word
            ners.append(ent)
            ent_vec = parsed[ent_id]
            sent_vecs.append(ent_vec)
            entities.append(word)
            ent_ids.append(ent_id)
        if iob != 'B':
            sent_vecs[-1] += parsed[ent_id]
            entities[-1] += ' ' + word
    return sent_vecs, ners, entities, ent_ids


def build_relation_data(ner_vecs, threshold=10):
    possible_relations = {}
    word_relations = {}
    for sent_id, ner_list in ner_vecs.items():
        ner_dict = defaultdict(list)
        for vec, ner_tag, entity, ent_id in zip(ner_list[0], ner_list[1], ner_list[2], ner_list[3]):
            ner_dict[ner_tag] += [(vec, entity, ent_id)]
        sent_relations = {}
        word_format = {}
        for rel_type, ner_tags in RELATION_TYPES.items():
            possible_ent1, possible_ent2 = ner_tags
            if any(ent in ner_dict for ent in possible_ent1) and any(ent in ner_dict for ent in possible_ent2):
                sent_relations[rel_type] = [(vec1, vec2) for ent1 in possible_ent1 for vec1 in ner_dict[ent1]
                                            for ent2 in possible_ent2 for vec2 in ner_dict[ent2]
                                            if distance_between_ents(vec1[2], vec2[2]) < threshold]
                word_format[rel_type] = [(vec1[1], vec2[1]) for (vec1, vec2) in sent_relations[rel_type]]
        possible_relations[sent_id] = sent_relations
        word_relations[sent_id] = word_format
    return possible_relations, word_relations


def distance_between_ents(ent1, ent2):
    return int(math.fabs(ent1 - ent2))


def tag_possible_relations(gold_relations, possible_relations, bad_examples=0.3):
    train_set = []
    train_tags = []
    for sent_id, relations_dict in possible_relations.items():
        for rel_type, entity_list in relations_dict.items():
            for (ent1, ent2) in entity_list:
                r = random.randint(0, 9) / 10.0
                if any([ent1[1], ent2[1]] == relation for relation in gold_relations[sent_id][rel_type]):
                    train_set.append(np.concatenate([ent1[0], ent2[0]]))
                    train_tags.append(1)
                elif r < bad_examples:
                    train_set.append(np.concatenate([ent1[0], ent2[0]]))
                    train_tags.append(0)
    return train_set, train_tags


if __name__ == '__main__':
    d = new_load_data('Corpus/train.txt')
    nlp = spacy.load('en')
    ner_vecs = {sent_id: get_new_entity_vecs(nlp, sentence) for sent_id, sentence in d.items()}
    print('yay')