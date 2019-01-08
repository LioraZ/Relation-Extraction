import codecs 
import spacy 
import sys
import random
from sklearn.utils import shuffle
from sklearn.svm import SVC
import numpy as np

TRAIN = '/train.txt'
CORPUS = 'Corpus'

"""
def load_data(d_name):
    data = []
    vocab = set()
    nlp = spacy.load('en')
    with open(d_name, 'r') as file:
        for line in file:
            data.append(nlp(line))
            print (line)
            vocab.add(line.split(" "))
    return data, {w: i for i, w in enumerate(vocab)}
    
    if __name__ == '__main__':
    train_data, vocabulary = load_data(CORPUS + TRAIN) 
""" 
    
def sign(num):
    if(num>0):
        return 1
    if(num==0):
        return 0

    return -1
class SVM(object):

    def __init__(self,pi,lamda,epoch):
        self.w=np.zeros((1,768))
        self.Pi=pi
        self.lamda=lamda
        self.epoch=epoch
        
    def execute(self,x,y):
        beginPi=self.Pi
        for t in range(1,self.epoch):
            self.Pi=beginPi/np.sqrt(t)
            x,y = shuffle(x,y)
            for pic,tag in zip(x,y):
                if(tag == -1 and (tag*np.dot(self.w,pic) < 0)):
                #    print ("in")
                    continue
                if(1-tag*np.dot(self.w,pic) > 0):
                    self.w=(1-self.Pi*self.lamda)*self.w+pic*self.Pi*tag
                else:
                    self.w=(1-self.Pi*self.lamda)*self.w

    def predict(self,test_x,test_y):
        counter=0
        ls = []
        for i,ex in enumerate(test_x):
            #print (sign(np.dot(self.w,ex)))
            ls.append(sign(np.dot(self.w, ex)))
            if(sign(np.dot(self.w,ex)) == test_y[i]):

                counter+=1
        print("accuracy is: "+str((float(counter)/len(test_y))*100)+"%")
        print(ls.count(1))
        return ls


    #predicts value for a single example
    def singlePredict(self,ex):
        return sign(np.dot(self.w,ex))


def read_lines(fname):
    for line in codecs.open(fname, encoding="utf8"):
        sent_id, sent = line.strip().split("\t")
        sent = sent.replace("-LRB-","(")
        sent = sent.replace("-RRB-",")")
        yield sent_id, sent

def get_relations_per_sen(fname):
    rel_per_sen = []
    prev_sen_idx = "sent10"
    curr_sen = []
    with open(fname, 'r') as file:
        for line in file:
            ls = line.split("\t")
            rel = ls[1:4]
            if (prev_sen_idx == ls[0]):
                curr_sen.append(rel)  
            else:
                rel_per_sen.append(curr_sen)
                curr_sen = [rel]
                prev_sen_idx = ls[0]
    rel_per_sen.append(curr_sen)
    return rel_per_sen
    


def get_ners_per_sen1(fname):
    flag = 0
    ner_per_sen = []
    nlp = spacy.load('en')
    for sent_id, sent_str in read_lines(fname):
        ners = []
        #print (sent_str)
        sent1 = nlp(sent_str)
        #print ("#id:",sent_id)
        #print ("#text:",sent.text)
        sent = (sent1)
        for ne in sent.ents:  
            ners.append((ne.root, ne.text))
        ner_per_sen.append(ners)              
    return (ner_per_sen)



def get_ners_per_sen(fname):
    flag = 0
    ner_per_sen = []
    nlp = spacy.load('en')
    for sent_id, sent_str in read_lines(fname):
        ners = []
        #print (sent_str)
        sent1 = nlp(sent_str)
        #print ("#id:",sent_id)
        #print ("#text:",sent.text)
        sent = (sent1)
        strr =""
        for word in sent:   
            if (word.ent_type_ in ['LOC', 'GOP', 'PERSON', 'GPE', 'ORG', 'FAC', 'NORP', 'QUANTITY', 'EVENT']):
                if (word.ent_iob_ == 'B'):
                    flag = 1
                    strr += word.text
                    #print((sent.ents[0]).vector)
                    ner = word
                elif (flag == 1 and word.ent_iob_ == 'I'):
                    strr += (" " + word.text)
            elif (flag == 1):
                ners.append((ner, strr))
                flag = 0
                strr = ""
        ner_per_sen.append(ners)              
    return (ner_per_sen)

def check_if_tuple_in_rel(rel, tpl):
    #print (rel)
    #print (tpl)
    if (rel[0].replace(" ", "") == tpl[0].replace(" ", "") and rel[2].replace(" ", "") == tpl[1].replace(" ", "") and rel[1] == "Live_In"):
        return True
    elif(rel[0].replace(" ", "") == tpl[1].replace(" ", "") and rel[2].replace(" ", "") == tpl[0].replace(" ", "") and rel[1] == "Live_In"): 
        return True
    return False

def get_train_data(ano_file, reg_file, flag = -1):
    data_x = []
    data_y = []
    sens_rels = get_relations_per_sen(ano_file) if flag == 1 else get_relations_per_sen(ano_file)[1:]
    sens_ners = get_ners_per_sen1(reg_file)
    #print("=================", len(sens_rels), len(sens_ners))
    for rels, ners in zip(sens_rels, sens_ners):
        #print ("=================================SEN=====================================", rels, ners)
        for rel in rels:
            #print ("====================================rel=============================")
            for idx, ner in enumerate(ners):
                for i in range(idx + 1, len(ners)):
                    exm = np.concatenate((ner[0].vector , ners[i][0].vector))
                    rnd = random.uniform(0, 1)
                    if (check_if_tuple_in_rel(rel, [ner[1], ners[i][1]])):
                        data_x.append(exm)                    
                        data_y.append(1)
                    elif (rel[1] == "Live_In"):
                        data_x.append(exm)                    
                        data_y.append(-1)
                    elif (rnd < 0.1 or flag == -1):
                        data_x.append(exm)                    
                        data_y.append(-1)
    return data_x, data_y
"""                          
def get_test_data():
    data_x = []
    data_y = []
    sens_rels = get_relations_per_sen("Annotations/dev.txt")
    #print (sens_rels)
    sens_ners = get_ners_per_sen('Corpus/dev.txt')
    #print ()
    for rels, ners in zip(sens_rels, sens_ners):
        for rel in rels:
            for idx, ner in enumerate(ners):
                for i in range(idx + 1, len(ners)):
                    data_x.append(ner[0].vector + ners[i][0].vector)
                    if (check_if_tuple_in_rel(rel, [ner[1], ners[i][1]])):
                        data_y.append(1)
                    else:
                        data_y.append(0)
    return data_x, data_y    
"""                       
    
#print (get_train_data()[1].count(1))
    
    
x, y = get_train_data("Annotations/train.txt", 'Corpus/train.txt', 1)
nlp = spacy.load('en')
print (str(y.count(1)) + " / " + str(len(y)))
tx,ty = get_train_data("Annotations/dev.txt", 'Corpus/dev.txt')
print (str(ty.count(1)) + " / " + str(len(ty)))
svm = SVM(0.15,0.1, 60)
svm.execute(x, y)
svm.predict(tx,ty)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        

   

