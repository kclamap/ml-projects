from __future__ import print_function
import argparse
import json
import os
import time
from keras.models import Model, load_model
from keras.layers import Input, Dense, Embedding, Dropout, TimeDistributed, BatchNormalization, ThresholdedReLU, Activation, Add
from keras.layers import LSTM, GRU
from keras.optimizers import Adam
from keras import initializers, constraints, regularizers
from keras.engine.topology import Layer
from keras import backend as K
from keras.callbacks import EarlyStopping
import numpy as np
from data_helper import load_data, build_input_data
from scorer import scoring
from utils import TestCallback, make_submission

from keras_self_attention import SeqSelfAttention
from keras.layers.core import *
#from keras.layers.merge import Merge




    

def build_model(embedding_dim, hidden_size, activation, drop, learning_rate, lr_decay, sequence_length, vocabulary_size):
    inputs = Input(shape=(sequence_length,), dtype='int32')
    # inputs -> [batch_size, sequence_length]

    emb_layer = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)
    # emb_layer.trainable = False
    # if you uncomment this line, the embeddings will be untrainable

    embedding = emb_layer(inputs)
    embedding = TimeDistributed(BatchNormalization())(embedding)
    # embedding -> [batch_size, sequence_length, embedding_dim]

    #drop_embed = Dropout(drop)(embedding)
    # dropout at embedding layer
    
    # add a LSTM here, set units=hidden_size, dropout=drop, recurrent_dropout = drop, return_sequences=True
    # please read https://keras.io/layers/recurrent/
    lstm_out_1 = LSTM(units=hidden_size, dropout=0, activation=activation, recurrent_activation="sigmoid", recurrent_dropout=drop, return_sequences=True)(embedding)
    lstm_out_1 = TimeDistributed(BatchNormalization())(lstm_out_1)
    #lstm_out_1 = Dropout(drop)(lstm_out_1)
    #gru_out_1 = GRU(units = hidden_size, activation=activation, dropout=drop, recurrent_dropout=drop, return_sequences = True, kernel_regularizer=regularizers.l1_l2(0), activity_regularizer=regularizers.l1(0))(BatchNormalization()(drop_embed))
    # lstm_out_1 -> [batch_size, sequence_length, hidden_size]
    
    lstm_out_2 = LSTM(units=hidden_size, dropout=drop, activation=activation, recurrent_activation="sigmoid", recurrent_dropout=drop, return_sequences=True)(lstm_out_1)
    lstm_out_2 = TimeDistributed(BatchNormalization())(lstm_out_2)
    lstm_out_2 = Dropout(drop)(lstm_out_2)
    #gru_out_2 = GRU(units = hidden_size, activation=activation, dropout=drop, recurrent_dropout=drop, return_sequences = True, kernel_regularizer=regularizers.l1_l2(0), activity_regularizer=regularizers.l1(0))(BatchNormalization()(gru_out_1))
    # lstm_out_1 -> [batch_size, sequence_length, hidden_size]

    #lstm_out_3 = LSTM(units=hidden_size, dropout=drop, activation=activation, recurrent_activation="sigmoid", recurrent_dropout=drop, return_sequences=True)(BatchNormalization()(lstm_out_2))
    #gru_out_3 = GRU(units = hidden_size, activation=activation, dropout=drop, recurrent_dropout=drop, return_sequences = True, kernel_regularizer=regularizers.l1_l2(0), activity_regularizer=regularizers.l1(0))(BatchNormalization()(gru_out_2))

    #gru_out_4 = GRU(units = hidden_size, activation=activation, dropout=drop, recurrent_dropout=drop, return_sequences = True, kernel_regularizer=regularizers.l1_l2(0), activity_regularizer=regularizers.l1(0))(BatchNormalization()(gru_out_3))

    # add a TimeDistributed here, set units=hidden_size, dropout=drop, recurrent_dropout = drop, return_sequences=True
    # please read  https://keras.io/layers/wrappers/
    # output: outputs -> [batch_size, sequence_length, vocabulary_size]

    #attention = SeqSelfAttention(attention_activation='sigmoid', kernel_regularizer=regularizers.l2(5e-2), bias_regularizer=regularizers.l1(1e-1), attention_regularizer_weight=1e-1)(BatchNormalization()(lstm_out_2))
    attention = Dense(1, activation='tanh')(lstm_out_2)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(hidden_size)(attention)
    attention = Permute([2, 1])(attention)

    attention_out = Add()([lstm_out_1, attention])

    dense_out_1 = TimeDistributed(Dense(units = hidden_size*12))(attention_out)
    dense_out_1 = TimeDistributed(Activation('elu'))(dense_out_1)
    dense_out_1 = Dropout(drop)(dense_out_1)

    
    outputs = TimeDistributed(Dense(units=vocabulary_size, activation='softmax'))(dense_out_1)

    # End of Model Architecture
    # ----------------------------------------#

    model = Model(inputs=inputs, outputs=outputs)

    adam = Adam(lr=learning_rate, decay=lr_decay)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam)

    print(model.summary())
    return model


def predict_final_word(model, vocabulary, filename):
    id_list = []
    prev_tokens_list = []
    prev_tokens_lens = []
    with open(filename, "r") as fin:
        fin.readline()
        for line in fin:
            id_, prev_sent, grt_last_token = line.strip().split(",")
            id_list.append(id_)
            prev_tokens = prev_sent.split()
            prev_tokens_list.append(prev_tokens)
            prev_tokens_lens.append(len(prev_tokens))
    X = np.array([build_input_data(t, vocabulary)[0][0].tolist()
                  for t in prev_tokens_list])
    y_prob = model.predict(X, batch_size=32)
    last_token_probs = np.array([y_prob[b, prev_tokens_lens[b] - 1, :]
                                 for b in range(y_prob.shape[0])])

    return dict(zip(id_list, last_token_probs))


def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    if opt.mode == "train":
        st = time.time()
        print('Loading data')
        x_train, y_train, x_valid, y_valid, vocabulary_size = load_data(
            "data", opt.debug)

        num_training_data = x_train.shape[0]
        sequence_length = x_train.shape[1]
        print(num_training_data)

        print('Vocab Size', vocabulary_size)

        model = build_model(opt.embedding_dim, opt.hidden_size, opt.activation, opt.drop, opt.learning_rate, opt.lr_decay, sequence_length, vocabulary_size)
        print("Traning Model...")

        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
        
        history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=opt.batch_size,
                            epochs=opt.epochs, verbose=1,
                            callbacks=[TestCallback((x_valid,y_valid), model=model), early_stop], shuffle = True)
        model.save(opt.saved_model)
        print("Training cost time: ", time.time() - st)

    else:
        model = load_model(opt.saved_model, custom_objects={'SeqSelfAttention': SeqSelfAttention})
        vocabulary = json.load(open(os.path.join("data", "vocab.json")))
        predict_dict = predict_final_word(model, vocabulary, opt.input)
        sub_file = make_submission(predict_dict, opt.student_id, opt.input)
        if opt.score:
            scoring(sub_file, os.path.join("data"), type="valid")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default="train", choices=["train", "test"],
                        help="Train or test mode")
    parser.add_argument("-saved_model", type=str, default="model.h5",
                        help="saved model path")
    parser.add_argument("-input", type=str, default=os.path.join("data", "valid.csv"),
                        help="Input path for generating submission")
    parser.add_argument("-debug", action="store_true",
                        help="Use validation data as training data if it is true")
    parser.add_argument("-score", action="store_true",
                        help="Report score if it is")
    parser.add_argument("-student_id", default=None, required=True,
                        help="Student id number is compulsory!")

    parser.add_argument("-epochs", type=int, default=1,
                        help="training epoch num")
    parser.add_argument("-batch_size", type=int, default=32,
                        help="training batch size")
    parser.add_argument("-embedding_dim", type=int, default=100,
                        help="word embedding dimension")
    parser.add_argument("-hidden_size", type=int, default=500,
                        help="rnn hidden size")
    parser.add_argument("-drop", type=float, default=0.5,
                        help="dropout")
    parser.add_argument("-activation", type=str, default="tanh",
                        help="dropout")
    parser.add_argument("-learning_rate", type=float, default=0.001,
                        help="dropout")
    parser.add_argument("-lr_decay", type=float, default=0.001,
                        help="dropout")
    parser.add_argument("-gpu", type=str, default="",
                        help="dropout")
    opt = parser.parse_args()
    main(opt)
