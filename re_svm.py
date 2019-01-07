import spacy
import pickle
from sklearn.svm import SVC
import utils


def train_svm(processed_data):
    ner_vecs = {sent_id: utils.get_entity_vecs(nlp, sentence) for sent_id, sentence in processed_data.items()}
    gold_relations = utils.get_gold_relations('Annotations/train.txt')
    possible_relations, _ = utils.build_relation_data(ner_vecs)
    train_set, train_tags = utils.tag_possible_relations(gold_relations, possible_relations)
    svm = SVC()
    svm.fit(train_set, train_tags)
    return svm


if __name__ == '__main__':
    pr_data = utils.get_processed_data('Proccessed/test.txt')
    nlp = spacy.load('en')
    model = train_svm(pr_data)
    pickle.dump(model, open('model_file', 'wb'))


