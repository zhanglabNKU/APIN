#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 16:12:37 2019

@author: joy1314bubian
"""

import numpy as np
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.optimizers import *
import argparse
import os



def seq_to_num(line,seq_length):
    seq = np.zeros(seq_length)
    for j in range(len(line)):
        seq[seq_length - 1 - j] = protein_dict[line[len(line)-j-1]]
    return seq





def get_word_freq(data):
    word_bag = []
    for seq in data:
        bag = np.zeros(21)
        for word in seq:
            bag[int(word)] = bag[int(word)] + 1
        bag = np.delete(bag,0)
        bag = bag/np.sum(bag)
        word_bag.append(bag)
    return word_bag


def getDepetideIndex(seq,index):
    return int((seq[index]-1)*20 + seq[index+1] - 1)

def get_depeptide_freq(data):
    freqs = []
    for seq in data:
        depeptide_freq = np.zeros(400)
        for i in range(len(seq)-1):
            if seq[i] != 0:
                index = getDepetideIndex(seq,i)
                depeptide_freq[index] = depeptide_freq[index] + 1
        depeptide_freq = depeptide_freq/np.sum(depeptide_freq)
        freqs.append(depeptide_freq)
    return freqs


def normalization_layer(input_layer):
    output_layer = Lambda(lambda x: x - K.mean(x))(input_layer)
    output_layer = Lambda(lambda x: x / K.max(x))(input_layer)
    return output_layer

def training(epochs,attention_num,attention_range,X_train,Y_train,X_test):
    inputs_bag = Input(shape = (420,))
    combine_layer = Dense(64, activation='sigmoid')(inputs_bag)
    inputs_normal = Input(shape = (max_length,))
    embedding_layer = Embedding(output_dim=embed_length, input_dim=21, input_length=max_length)(inputs_normal)
    attention_layers = []
    for i in range(attention_range):
        attention_layer = Conv1D(attention_num, (i+1)*2, strides=1,padding="same", activation='relu')(embedding_layer)
        pooling_layer = MaxPooling1D(pool_size = max_length)(attention_layer)
        attention_layers.append(pooling_layer)
    cov_layer = Concatenate(axis = 1)(attention_layers)
    cov_layer = MaxPooling1D(pool_size = attention_range)(cov_layer)
    cov_layer = Reshape((attention_num,))(cov_layer)
    cov_layer = Concatenate()([normalization_layer(combine_layer),normalization_layer(cov_layer)])
    cov_layer =Dropout(0.2)(cov_layer)
    output = Dense(1, activation='sigmoid')(cov_layer)
    model = Model(inputs = [inputs_bag,inputs_normal], outputs=output)
    rms = RMSprop(lr=0.001)
    model.compile(optimizer=rms,loss='binary_crossentropy',metrics=['accuracy'])
    fit = model.fit([X_train_other,X_train], Y_train , epochs= epochs, batch_size = 32)
    res = model.predict([X_test_other,X_test], batch_size = 128)
    return res

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='proposed fusion model')

    parser.add_argument('-epochs', default=10, type=int)
    parser.add_argument('-attention_num', default=64, type=int)
    parser.add_argument('-attention_range', default=14, type=int)
    parser.add_argument('-embed_length', default=128, type=int)
    parser.add_argument('-max_length', default=200, type=int)
    parser.add_argument('-seq_length', default=200, type=int)
    parser.add_argument('-prediction_file',default='proposed_fusion_prediction_output.txt',type=str)
    parser.add_argument('-true_train_file', default='data/AMP.tr.fa', type=str)
    parser.add_argument('-false_train_file', default='data/DECOY.tr.fa', type=str)
    parser.add_argument('-test_file', default='data/AMP.te.fa', type=str)   
    args = parser.parse_args()
    attention_num = args.attention_num
    attention_range=args.attention_range
    embed_length=args.embed_length
    max_length = args.max_length
    seq_length = args.seq_length
    epochs=args.epochs
    prediction_file=args.prediction_file
    test_file=args.test_file
    train_true_file=args.true_train_file
    train_false_file=args.false_train_file
    protein_dict = {'Z':0, 
                'A':1, 
                'C':2, 
                'D':3, 
                'E':4, 
                'F':5, 
                'G':6, 
                'H':7, 
                'I':8,
                'K':9, 
                'L':10, 
                'M':11, 
                'N':12, 
                'P':13, 
                'Q':14, 
                'R':15, 
                'S':16, 
                'T':17, 
                'V':18, 
                'W':19, 
                'Y':20}
    X_test = []
    file =open(test_file,'r')
    text = []
    read_text = file.readlines()
    file.close()
    text.extend(read_text)
    for i in range(len(text)//2):
        line = text[i*2+1]
        line = line[0:len(line)-1]
        seq = seq_to_num(line,seq_length)
        X_test.append(seq)
    X_train = []
    file =open(train_true_file,'r')
    text = []
    read_text = file.readlines()
    file.close()
    text.extend(read_text)

    file =open(train_false_file,'r')
    read_text = file.readlines()
    file.close()
    text.extend(read_text)
    for i in range(len(text)//2):
        line = text[i*2+1]
        line = line[0:len(line)-1]
        seq = seq_to_num(line,seq_length)
        X_train.append(seq)
    Y_train = (np.zeros(len(X_train)//2) + 1).tolist()
    Y_train.extend(np.zeros(len(X_train)//2).tolist())
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    Y_train=np.array(Y_train)
    X_train_depeptide = np.array(get_depeptide_freq(X_train))
    X_test_depeptide = np.array(get_depeptide_freq(X_test))
    X_train_bag = np.array(get_word_freq(X_train))
    X_test_bag = np.array(get_word_freq(X_test))
    X_train_other = np.concatenate([X_train_depeptide,X_train_bag],axis = 1)
    X_test_other = np.concatenate([X_test_depeptide,X_test_bag],axis = 1)

    pred_label=training(epochs,attention_num,attention_range,X_train,Y_train,X_test)
    current_path=os.getcwd()
    dir_list=os.listdir(current_path)
    if 'output' not in dir_list:
        os.mkdir('output')
    f=open('output/'+prediction_file,'w')
    for i in range(len(pred_label)):
        if pred_label[i][0]>=0.5:
            f.write('1\n')
        else:
            f.write('0\n')
    f.close()