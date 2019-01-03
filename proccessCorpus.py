import spacy
from train_bilstm import BI_LSTM, RE_MLP
import dynet as dy
import itertools

LAYERS = 1
HIDDEN_DIM = 100
EMBED_DIM = 384


TRAIN = '/train.txt'
CORPUS = 'Corpus'


def load_data(d_name):
    data = []
    vocab = set()
    nlp = spacy.load('en')
    with open(d_name, 'r') as file:
        for line in file:
            data.append(nlp(line))
            vocab.add(line.split())
    return data, {w: i for i, w in enumerate(vocab)}


def get_relations_from_sentence(sentence):
    loc_entities = [entity for entity in sentence.ents if entity.label in ['LOC', 'GOP']]
    per_entities = [entity for entity in sentence.ents if entity.label in ['PERSON']]
    return loc_entities, per_entities


def get_trained_embeds(data):
    sentence_dict = {}
    model = dy.Model()
    re_model = dy.Model()
    tagger = BI_LSTM(model, LAYERS, EMBED_DIM, HIDDEN_DIM, len(vocabulary))
    relation_extractor = RE_MLP(re_model, EMBED_DIM, HIDDEN_DIM)

    for sentence in data:
        if len(sentence.ent) < 2:
            continue
        good = bad = 0.0
        sentence_idxs = [vocabulary[word] for word in sentence.doc.split()]
        encodings = tagger.train_sentence(sentence, sentence_idxs)
        location_entities, person_entities = get_relations_from_sentence(sentence)
        cartesian_product = itertools.product([person_entities, location_entities])
        for relation in cartesian_product:
            is_relation = (relation in relation_dict[sentence_num])
            pred_relation = relation_extractor.train_relation(relation, is_relation)
            good += 1 if is_relation == pred_relation else 0
            bad += 1 if is_relation != pred_relation else 0


def read_annotations():
    print(yay)
    return relations


if __name__ == '__main__':
    train_data, vocabulary = load_data(CORPUS + TRAIN)
    get_trained_embeds(train_data)