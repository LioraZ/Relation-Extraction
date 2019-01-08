import spacy
import pickle
from sklearn.svm import SVC, libsvm
import utils
import numpy as np
from sklearn.utils import shuffle


def sign(num):
    if (num > 0):
        return 1
    if (num == 0):
        return 0

    return -1


class SVM(object):
    def __init__(self, pi, lamda, epoch):
        self.w = np.zeros((1, 768))
        self.Pi = pi
        self.lamda = lamda
        self.epoch = epoch

    def execute(self, x, y):
        beginPi = self.Pi
        for t in range(1, self.epoch):
            self.Pi = beginPi / np.sqrt(t)
            x, y = shuffle(x, y)
            for pic, tag in zip(x, y):
                if (tag == -1 and (tag * np.dot(self.w, pic) < 0)):
                    #    print ("in")
                    continue
                if (1 - tag * np.dot(self.w, pic) > 0):
                    self.w = (1 - self.Pi * self.lamda) * self.w + pic * self.Pi * tag
                else:
                    self.w = (1 - self.Pi * self.lamda) * self.w

    def predict(self, test_x):
        ls = []
        for i, ex in enumerate(test_x):
            # print (sign(np.dot(self.w,ex)))
            ls.append(sign(np.dot(self.w, ex)))
        return ls

    # predicts value for a single example
    def singlePredict(self, ex):
        return sign(np.dot(self.w, ex))



def train_svm(processed_data):
    ner_vecs = {sent_id: utils.get_entity_vecs(nlp, sentence) for sent_id, sentence in processed_data.items()}
    gold_relations = utils.get_gold_relations('Annotations/train.txt')
    possible_relations, _ = utils.build_relation_data(ner_vecs)
    train_set, train_tags = utils.tag_possible_relations(gold_relations, possible_relations)
    svm = SVC(kernel='rbf')
    svm.fit(train_set, train_tags)
    # svm = SVM(0.15,0.1, 60)
    # svm.execute(train_set, train_tags)
    return svm


if __name__ == '__main__':
    pr_data = utils.get_processed_data('Proccessed/train.txt')
    nlp = spacy.load('en')
    model = train_svm(pr_data)
    pickle.dump(model, open('model_file', 'wb'))


