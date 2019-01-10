import spacy
import itertools
import pickle
import sys
from sklearn.svm import SVC
from re_svm import SVM
from spacy import displacy
import nltk
import utils
import numpy as np


np_grammar = r"""
    NP:
    {(<NN|NNS>|<NNP|NNPS>)<NNP|NN|NNS|NNPS>+}
    {(<NN|NNS>+|<NNP|NNPS>+)<IN|CC>(<PRP\$|DT><NN|NNS>+|<NNP|NNPS>+)}
    {<JJ|RB|CD>*<NNP|NN|NNS|NNPS>+}
    {<NNP|NN|NNS|NNPS>+}
    CD:
    {<CD>+}
    """



def get_dev_data(possible_relations):
    return [np.concatenate([ent1[0], ent2[0]]) for sent_ents in possible_relations.values() for rel_type, relation_type_list
            in sent_ents.items() for (ent1, ent2) in relation_type_list]


def predict_dev(svm, processed_data):
    ner_vecs = {sent_id: utils.get_entity_vecs(nlp, sentence) for sent_id, sentence in processed_data.items()}
    t = ner_vecs['sent1656']
    possible_relations, word_format = utils.build_relation_data(ner_vecs)
    dev_set = get_dev_data(possible_relations)
    predictions = svm.predict(dev_set)
    return predictions, word_format


def has_relation(annotation, processed_data):
    sent_id, ent1, rel_type, ent2 = annotation
    if sent_id == 'sent1838':
        print('yay')
        sent_tokens = processed_data[sent_id][utils.SENTENCE]
        loc_kw = ['from', 'in']
        # nltk_parse = nltk.ChartParser(sent_tokens)
        doc = nlp(' '.join(sent_tokens))
    if rel_type == 'Live_In':
        np_parser = nltk.RegexpParser(np_grammar)
        # keywords: of, New Jersey Gov. Thomas Kean, is from, Located_In relations, in (mainly for Located_In relation),
        # Texas Agriculture Commissioner Jim Hightower, Bush..... in U.S., Wang Shaohua , an official with China 's consulate in San Francisco
        # Hua Wen-Yi , a famous opera singer in Shanghai, said Leonard Lee , editor of the Chinese Times , a Chinese language daily newspaper in San Francisco
        # Marie Magdefrau Ferraro , 50 , of Bethany, Conn, David, of the.... in Ohio, David Leahy , elections supervisor for Dade County,
        # said Robert Isaacks , an emergency medical technician on High Island,
        sent_tokens = processed_data[sent_id][utils.SENTENCE]
        loc_kw = ['from', 'in']
        nltk_parse = nltk.ChartParser(np_grammar)

        doc = nlp(' '.join(sent_tokens))
        for w in doc:
            if w.dep_ == 'ROOT':
                print_dep_tree(w)
                break
        #print(nltk_parse.parse([(w.text, w.tag_) for w in doc]))
        root1 = root2 = None
        for chunk in doc.noun_chunks:
            if ent1 in chunk.text and ent2 in chunk.text:
                return True
            if ent1 in chunk.text and chunk.root.head.text == 'is':
                root1 = chunk.root.head.text
            if ent2 in chunk.text and chunk.root.head.text in loc_kw:
                root2 = chunk.root.head.text
            print(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text)
        if root1 is not None and root2 is not None:
            if root1 == 'is' and root2 in loc_kw:
                return True
        print(sent_tokens)

    return False


def print_dep_tree(root, tabs=''):
    for c in root.lefts:
        print_dep_tree(c, tabs + '\t')
    print(tabs + root.text)
    for c in root.rights:
        print_dep_tree(c, tabs + '\t')


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