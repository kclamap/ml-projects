import numpy as np
import os
import nltk
import keras.backend as K
from keras.models import Model
from keras.layers import Bidirectional, Input, Dense, Activation, Embedding, Dropout, TimeDistributed, GRU, Add, Lambda
from keras.layers import dot, concatenate
from keras.optimizers import Adam
from keras.callbacks import Callback
from data_helper import load_data

# os.environ['CUDA_VISIBLE_DEVICES']="" # uncomment this line, if you use cpu

# training parameters
dropout_rate = 0.2
epochs = 20
batch_size = 64
embedding_dim = 100

# gru parameters
hidden_dim = 100
num_encoder_layer = 2
num_decoder_layer = 1

# attention parameters
attention_type = 'dot'  # [None, 'dot', 'multiplicative', 'additive']


def seq2seq_predict(seq2seq_model, encoder_input, decoder_sequence_length, sos_idx):
    # because we do not have the truth decoder input,
    # we need to use the decoder prediction as its input
    decoder_input = np.zeros(
        shape=(len(encoder_input), decoder_sequence_length))
    decoder_input[:, 0] = sos_idx
    for i in range(1, decoder_sequence_length):
        output = seq2seq_model.predict(
            [encoder_input, decoder_input], batch_size=batch_size).argmax(axis=2)
        decoder_input[:, i] = output[:, i]
    decoder_output = decoder_input
    return decoder_output


def recover_sentence(x, idx2word):
    s = []
    for idx in x:
        word = idx2word[idx]
        if word == '<sos>':
            continue
        elif word == '<eos>':
            break
        elif word == '<pad>':
            break
        s.append(word)
    return s


class TestCallback(Callback):
    """
    Calculate BLEU
    """

    def __init__(self, test_data, model, vocabulary):
        self.test_data = test_data
        self.model = model
        self.vocabulary = vocabulary
        self.idx2word = dict()
        for k, v in self.vocabulary.items():
            self.idx2word[v] = k

    def on_epoch_end(self, epoch, logs={}):
        [encoder_input, decoder_input, decoder_target] = self.test_data
        decoder_output = seq2seq_predict(
            self.model, encoder_input, decoder_input_train.shape[1], vocabulary['<sos>'])
        bleu, results = self.evaluate_bleu(decoder_target, decoder_output)
        results.sort(reverse=True)
        print('Validation Set BLEU: %f' % (bleu))
        print('Top | BLEU | %s | %s' %
              ('target'.ljust(20), 'output'.ljust(20)))
        indices = list(range(len(results)))
        candidate_indices = list()
        candidate_indices.extend(indices[0:3])
        step = len(indices)//10
        if step > 0:
            candidate_indices.extend(indices[2+step::step])
        if indices[-1] != candidate_indices[-1]:
            candidate_indices.append(indices[-1])
        for i in candidate_indices:
            r = results[i]
            print('%-4d|%.4f| %s | %s' %
                  (i, r[0], ' '.join(r[1]), ' '.join(r[2])))

    def evaluate_bleu(self, target, output):
        N = target.shape[0]
        sum_bleu = 0.0
        results = []
        for i in range(N):
            t = recover_sentence(target[i], self.idx2word)
            o = recover_sentence(output[i], self.idx2word)
            bleu = nltk.translate.bleu_score.sentence_bleu([t], o)
            sum_bleu += bleu
            results.append((bleu, t, o))
        return sum_bleu / N, results


if __name__ == '__main__':
    print('Loading data')
    encoder_input_train, decoder_input_train, decoder_target_train, \
        encoder_input_valid, decoder_input_valid, decoder_target_valid, vocabulary = load_data(
            'translation')
    vocab_size = len(vocabulary)

    print('encoder_input_train.shape', encoder_input_train.shape)
    print('decoder_input_train.shape', decoder_input_train.shape)
    print('Vocab Size', vocab_size)

    num_training_data = encoder_input_train.shape[0]
    encoder_sequence_length = encoder_input_train.shape[1]
    decoder_sequence_length = decoder_input_train.shape[1]

    # encoder_input -> [batch_size, encoder_sequence_length]
    # decoder_input -> [batch_size, decoder_sequence_length]
    encoder_input = Input(shape=(encoder_sequence_length,), dtype='int32')
    decoder_input = Input(shape=(decoder_sequence_length,), dtype='int32')

    # the encoder and decoder share the same embedding layer
    emb_layer = Embedding(input_dim=vocab_size,
                          output_dim=embedding_dim, mask_zero=True)

    ################
    # ENCODER PART #
    ################

    # embedding -> [batch_size, sequence_length, embedding_dim]
    ### YOUR CODE HERE ###
    encoder_input_embed = emb_layer(encoder_input)

    # dropout at embedding layer
    ### YOUR CODE HERE ###
    encoder_input_droped = Dropout(dropout_rate)(encoder_input_embed)

    # add multiple Bidirectional GRU layers here,
    # set units=hidden_dim, return_sequences=True at the previous layers
    # set units=hidden_dim, return_sequences=True, return_state=True at the last layer
    # please read https://keras.io/layers/recurrent/
    # output:
    #     if return_sequences==True:
    #         gru_output -> [batch_size, sequence_length, 2*hidden_dim]
    #     if return_sequences==True and return_state=True:
    #         gru_output -> [batch_size, sequence_length, 2*hidden_dim], [batch_size, hidden_dim], [batch_size, hidden_dim]

    # N − 1 layer(s) of Bidirectional GRU, which return(s) sequences only.
    # Dropout layers between GRU layers if applicable.
    encoder_inputs = [encoder_input_droped]
    for i in range(0, num_encoder_layer-1):
        ### YOUR CODE HERE ###
        encoder_output = Bidirectional(GRU(units = hidden_dim, return_sequences = True, unroll = True))(encoder_inputs[-1])
        encoder_output_droped = Dropout(dropout_rate)(encoder_output)
        encoder_inputs.append(encoder_output_droped)
    # 1 layer of Bidirectional GRU, which returns sequences and the last state.
    encoder_output, encoder_last_h, encoder_last_hr = Bidirectional(GRU(units=hidden_dim,
                                                                        return_sequences=True, 
                                                                        return_state=True, 
                                                                        unroll=True))(encoder_inputs[-1])

    ################
    # DECODER PART #
    ################

    # embedding -> [batch_size, sequence_length, embedding_dim]
    ### YOUR CODE HERE ###
    decoder_input_embed = emb_layer(decoder_input)

    # dropout at embedding layer
    ### YOUR CODE HERE ###
    decoder_input_droped = Dropout(dropout_rate)(decoder_input_embed)

    # add multiple Unidirectional GRU layers here,
    # set units=2*hidden_dim, return_sequences=True
    # set initial_state=encoder_hidden_state at the first layer
    # please read https://keras.io/layers/recurrent/
    # output:
    # gru_output -> [batch_size, sequence_length, 2*hidden_dim]

    # 1 layer of Bidirectional GRU, whose input is the output of the encoder’s last layer
    # and initial hidden state is the encoder’s last state.
    ### YOUR CODE HERE ###
    decoder_output = GRU(units = hidden_dim * 2, return_sequences = True, unroll = True)(decoder_input_droped, initial_state = concatenate([encoder_last_h, encoder_last_hr], axis = 1))
    decoder_outputs = [decoder_output]
    # M − 1 layer(s) of Bidirectional GRU.
    # Dropout layers between GRU layers if applicable.
    for i in range(1, num_decoder_layer):
        ### YOUR CODE HERE ###
        decoder_output_droped = Dropout(dropout_rate)(decoder_outputs[-1])
        decoder_output = GRU(units = hidden_dim * 2, return_sequences = True, unroll = True)(decoder_output_droped[-1])
        decoder_outputs.append(decoder_output)

    # simple seq2seq without any attention mechanism
    if attention_type is None:
        # 1 layer of Dense layer with softmax activation wrapped by TimeDistributed.
        output = TimeDistributed(
            Dense(units=vocab_size, activation='softmax'))(decoder_outputs[-1])
    else:
        # dot-product attention
        # weight_{i,j} = softmax(\sum_k {decoder_output_{i,k} * encoder_output_{j,k}})
        if attention_type == 'dot':
            ### YOUR CODE HERE ###
            weight = Activation('softmax')(dot([decoder_output, encoder_output], axes = [2, 2]))
        # multiplicative attention
        # weight_{i,j} = softmax(\sum_k {decoder_output_{i,k} * (W encoder_output_{j,k})})
        elif attention_type == 'multiplicative':
            ### YOUR CODE HERE ###
            weight = None
        # additive attention
        # weight_{i, j} = softmax(\sum_k {V tanh(W1 decoder_output_{i,k} + W2 encoder_output_{j,k})})
        elif attention_type == 'additive':
            ### YOUR CODE HERE ###
            # You may need the help of the Lambda wrapper(https://keras.io/layers/core/#lambda)
            weight = None
        else:
            raise NotImplementedError
        attention = dot([weight, encoder_output], axes=[2, 1])
        output = TimeDistributed(Dense(units=vocab_size, activation='softmax'))(
            concatenate([decoder_outputs[-1], attention], axis=2))

    model = Model(inputs=[encoder_input, decoder_input], outputs=[output])

    adam = Adam()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam)

    print(model.summary())

    print("Traning Model...")
    history = model.fit([encoder_input_train, decoder_input_train], np.expand_dims(decoder_target_train, axis=2), 
                        batch_size=batch_size, 
                        epochs=epochs,
                        verbose=1 if os.name == 'posix' else 2, 
                        callbacks=[TestCallback((encoder_input_valid, decoder_input_valid, decoder_target_valid), model, vocabulary)])
