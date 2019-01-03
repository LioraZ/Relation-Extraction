import dynet as dy
import numpy as np


class RE_MLP(object):
    def __init__(self, model, num_input, num_hidden, num_out=2):
        self.model = model
        HIDDEN_DIM = 100
        MLP_DIM = 100
        self.trainer = dy.AdagradTrainer(model, 0.01)
        self.W1 = model.add_parameter((num_out, HIDDEN_DIM))
        self.W2 = model.add_parameter((MLP_DIM, num_hidden * 2))
        self.pT = model.add_lookup_parameter((num_out, MLP_DIM))
        self.activation_func = dy.tanh
        self.spec = (num_input, num_hidden, num_out, self.activation_func)

    def __call__(self, x):
        W1 = dy.parameter(self.W1)
        W2 = dy.parameter(self.W2)
        return W2 * (self.activation_func(W1 * x))

    def train_relation(self, word_encoding, relation):
        dy.renew_cg()
        word1, word2 = word_encoding
        r_t = self(dy.concatenate([word1, word2]))
        temp_val = dy.softmax(r_t).value()
        is_relation = np.argmax(temp_val)
        error = dy.pickneglogsoftmax(r_t, relation)
        error.backward()
        self.trainer.update()
        return is_relation

    def dev_relation(self, word_encoding):
        dy.renew_cg()
        word1, word2 = word_encoding
        r_t = self(dy.concatenate([word1, word2]))
        temp_val = dy.softmax(r_t).value()
        is_relation = np.argmax(temp_val)
        return is_relation

    def test_sentence(self, words, word_idxs):
        dy.renew_cg()
        forward_init, backward_init = [b.initial_state() for b in self.builders]
        embed_words = words.tensor
        # entities = words.ents
        forward = forward_init.transduce(embed_words)
        backward = backward_init.transduce(reversed(embed_words))

        predictions = []
        for f, b in zip(forward, backward):
            r_t = self(dy.concatenate([f, b]))
            temp_val = dy.softmax(r_t).value()
            # chosen = np.argmax(temp_val)
            predictions.append(temp_val)

        return predictions

    # support saving:
    def param_collection(self): return self.model

    @staticmethod
    def from_spec(spec, model):
        num_layers, num_input, num_hidden, num_out = spec
        return BI_LSTM(model, num_input, num_hidden, num_out)


class BI_LSTM(object):
    def __init__(self, model, num_layers, num_input, num_hidden, num_out):
        self.model = model
        HIDDEN_DIM = 100
        MLP_DIM = 100
        self.trainer = dy.AdagradTrainer(model, 0.01)
        self.pH = model.add_parameter((num_out, HIDDEN_DIM))
        self.pO = model.add_parameter((MLP_DIM, num_hidden * 2))
        self.pT = model.add_lookup_parameter((num_out, MLP_DIM))
        self.builders = [dy.LSTMBuilder(num_layers, num_input, num_hidden, model),
                         dy.LSTMBuilder(num_layers, num_input, num_hidden, model)]
        self.activation_func = dy.tanh
        self.spec = (num_input, num_hidden, num_out, self.activation_func)

    def __call__(self, x):
        H = dy.parameter(self.pH)
        O = dy.parameter(self.pO)
        return O * (self.activation_func(H * x))

    def train_sentence(self, words, word_idxs):
        dy.renew_cg()
        forward_init, backward_init = [b.initial_state() for b in self.builders]
        embed_words = words.tensor
        # entities = words.ents
        forward = forward_init.transduce(embed_words)
        backward = backward_init.transduce(reversed(embed_words))

        errors = []
        encodings = []
        good = bad = 0.0
        for f, b, tag in zip(forward, backward, word_idxs):
            r_t = self(dy.concatenate([f, b]))
            temp_val = dy.softmax(r_t).value()
            chosen = np.argmax(temp_val)
            encodings.append(temp_val)
            good += 1 if chosen == tag else 0
            bad += 1 if chosen != tag else 0
            error = dy.pickneglogsoftmax(r_t, tag)
            errors.append(error)

        sum_errors = dy.esum(errors)
        loss = sum_errors.scalar_value()
        sum_errors.backward()
        self.trainer.update()
        accuracy = 100 * (good / (good + bad))
        print(str(accuracy), str(loss))

        return encodings

    def dev_sentence(self, words, word_idxs):
        dy.renew_cg()
        forward_init, backward_init = [b.initial_state() for b in self.builders]
        embed_words = words.tensor
        # entities = words.ents
        forward = forward_init.transduce(embed_words)
        backward = backward_init.transduce(reversed(embed_words))

        encodings = []
        good = bad = 0.0
        for f, b, tag in zip(forward, backward, word_idxs):
            r_t = self(dy.concatenate([f, b]))
            temp_val = dy.softmax(r_t).value()
            chosen = np.argmax(temp_val)
            encodings.append(temp_val)
            good += 1 if chosen == tag else 0
            bad += 1 if chosen != tag else 0

        accuracy = 100 * (good / (good + bad))
        print(str(accuracy))

        return encodings

    def test_sentence(self, words, word_idxs):
        dy.renew_cg()
        forward_init, backward_init = [b.initial_state() for b in self.builders]
        embed_words = words.tensor
        # entities = words.ents
        forward = forward_init.transduce(embed_words)
        backward = backward_init.transduce(reversed(embed_words))

        predictions = []
        for f, b in zip(forward, backward):
            r_t = self(dy.concatenate([f, b]))
            temp_val = dy.softmax(r_t).value()
            # chosen = np.argmax(temp_val)
            predictions.append(temp_val)

        return predictions

    # support saving:
    def param_collection(self): return self.model

    @staticmethod
    def from_spec(spec, model):
        num_layers, num_input, num_hidden, num_out = spec
        return BI_LSTM(model, num_layers, num_input, num_hidden, num_out)
