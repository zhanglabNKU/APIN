#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:13:51 2019

@author: joy1314bubian
"""


import numpy as np
import argparse
import os
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.optimizers import *



def training(epochs,attention_num,attention_range,X_train,Y_train,X_test):
    inputs_normal = Input(shape = (max_length,))
    embedding_layer = Embedding(output_dim=embed_length, input_dim=21, input_length=max_length)(inputs_normal)
    attention_layers = []
    for i in range(attention_range):
        attention_layer = Conv1D(attention_num, (i+1)*2, strides=1,padding="same", activation='relu')(embedding_layer)
        attention_layer = MaxPooling1D(pool_size = max_length)(attention_layer)       
        attention_layers.append(attention_layer)
    cov_layer = Concatenate(axis = 1)(attention_layers)
    cov_layer = MaxPooling1D(pool_size = attention_range)(cov_layer)
    cov_layer = Reshape((attention_num,))(cov_layer)
    cov_layer =Dropout(0.2)(cov_layer)
    output = Dense(1, activation='sigmoid')(cov_layer)
    model = Model(inputs = [inputs_normal], outputs=output)
    rms = RMSprop(lr=0.0002)
    model.compile(optimizer=rms,loss='binary_crossentropy',metrics=['accuracy'])
    fit = model.fit(X_train, Y_train , epochs= epochs, batch_size = 32)
    res = model.predict(X_test, batch_size = 128)
    return res





def seq_to_num(line,seq_length):
    seq = np.zeros(seq_length)
    for j in range(len(line)):
        seq[seq_length - 1 - j] = protein_dict[line[len(line)-j-1]]
    return seq





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='proposed model')

    parser.add_argument('-epochs', default=50, type=int)
    parser.add_argument('-attention_num', default=64, type=int)
    parser.add_argument('-attention_range', default=14, type=int)
    parser.add_argument('-embed_length', default=128, type=int)
    parser.add_argument('-max_length', default=200, type=int)
    parser.add_argument('-seq_length', default=200, type=int)
    parser.add_argument('-prediction_file',default='proposed_prediction_output.txt',type=str)
    parser.add_argument('-true_train_file', default='data/AMP.tr.fa', type=str)
    parser.add_argument('-false_train_file', default='data/DECOY.tr.fa', type=str)
    parser.add_argument('-test_file', default='data/test.fa', type=str)   
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
    pred_label=training(epochs,attention_num,attention_range,X_train,Y_train,X_test)
    current_path=os.getcwd()
    dir_list=os.listdir(current_path)
    if 'output' not in dir_list:
        os.mkdir('output')
    Y_pred=[]
    f=open('output/'+prediction_file,'w')
    for i in range(len(pred_label)):
        if pred_label[i][0]>=0.5:
            f.write('1\n')
            Y_pred.append(1)
        else:
            f.write('0\n')
            Y_pred.append(0)
    f.close()
    Y_test = (np.zeros(len(X_test)//2) + 1).tolist()
    Y_test.extend(np.zeros(len(X_test)//2).tolist())