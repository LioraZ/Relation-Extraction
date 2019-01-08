import spacy

ANNOTATION_TRAIN = 'Annotations/train.txt'
ANNOTATION_DEV = 'Annotations/dev.txt'


def get_sentence_by_relation(rel_type):
    d = []
    with open(ANNOTATION_TRAIN, 'r') as file:
        for line in file:
            fields = line.split('\t')
            if fields[2] == rel_type:
                d.append('\t'.join([fields[1], fields[3], fields[4]]))
    with open(ANNOTATION_DEV, 'r') as file:
        for line in file:
            fields = line.split('\t')
            if fields[2] == rel_type:
                d.append('\t'.join([fields[1], fields[3], fields[4]]))
    with open('all \'Live_In\' relations', 'w') as file:
        for item in d:
            file.write(item + '\n')


if __name__ == '__main__':
    nlp = spacy.load('en')
    get_sentence_by_relation('Live_In')