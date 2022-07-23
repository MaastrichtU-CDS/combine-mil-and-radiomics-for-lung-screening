# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:22:41 2020

@author: chenj
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
 

def lrelu(x,alpha=0.1):
    return tf.maximum(alpha*x,x)

def pares_tf(example_proto):
    dics={
            'RadiomicFeatures':tf.FixedLenFeature([], tf.string),
            'Label':tf.FixedLenFeature([], tf.string)
            }
    parsed_example = tf.parse_single_example(serialized=example_proto,features=dics)
    Radiomics= tf.decode_raw(parsed_example['RadiomicFeatures'],out_type=tf.float64)
    Radiomics = tf.reshape(Radiomics, [12,104])
    Label = tf.decode_raw(parsed_example['Label'],out_type=tf.int32)
    Label = tf.reshape(Label, [1])
    return Radiomics, Label

trainset=tf.data.TFRecordDataset(filenames=['./Radiomic features and an Example of TFrecord/SA60Train1.tfrecords'])
trainset=trainset.map(pares_tf)
trainset=trainset.shuffle(3000).repeat(502).batch(1)
iterator = trainset.make_one_shot_iterator()
next_patch = iterator.get_next()

testset=tf.data.TFRecordDataset(filenames=['./Radiomic features and an Example of TFrecord/SA60Test1.tfrecords'])
testset=testset.map(pares_tf)
testset=testset.shuffle(3000).repeat(11).batch(1)
iterator2 = testset.make_one_shot_iterator()
next_patch2 = iterator2.get_next()

inputs_ = tf.placeholder(tf.float32, [None, 104])
labels_ = tf.placeholder(tf.int32, [1])
def attention(inputs, attention_size, time_major=False, return_alphas=False):

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size =inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas
    
with tf.name_scope('MILnetwork'):
    fc1=tf.layers.dense(inputs_,
                        units =256,
                        activation=lrelu,
                        use_bias=True,
                        )
    dropout1=tf.nn.dropout(fc1,rate=0.5)
    fc2=tf.layers.dense(dropout1,
                        units =128,
                        activation=lrelu,
                        use_bias=True,
                        )
    dropout2=tf.nn.dropout(fc2,rate=0.5)
    fc3=tf.layers.dense(dropout2,
                        units =64,
                        activation=lrelu,
                        use_bias=True,
                        )
    dropout3=tf.nn.dropout(fc3,rate=0.5)
    embeding=tf.reshape(dropout3,[1,12,64])
    
    output,alphas=attention(inputs=embeding,attention_size=128,return_alphas=True)
    #output2=tf.reshape(output,[1,64])
    
    
    fc4=tf.layers.dense(output,
                           units=32,
                           activation=lrelu,
                           )
    fc5=tf.layers.dense(fc4,
                           units=2,
                           activation=tf.nn.softmax,
                           )
#pred_prob=tf.nn.softmax(result,axis=1)



Y=tf.one_hot(labels_, 2)

nb_classes = 2

W = tf.Variable(tf.random_normal([32,nb_classes]),name='weight')
b = tf.Variable(tf.random_normal([nb_classes]),name='bias')
#fc5_c = fc5.astype(np.float32)
hypothesis = tf.nn.softmax(tf.nn.xw_plus_b(fc4,W,b))
#hypothesis =np.max(tf.nn.softmax(fc5_c))

loss = tf.reduce_mean(tf.reduce_sum(abs(hypothesis-Y)*abs(hypothesis-Y),axis=1))


learning_rate = tf.placeholder(tf.float32)
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)


with tf.Session() as sess:
    epochs = 500
    batch_size = 1
    lr =1e-4
    AUC_final=[]
    precision_final=[]
    recall_final=[]
    PPV_final=[]
    NPV_final=[]
    for running in range(1):
        sess.run(tf.global_variables_initializer())
        total_batch=145
        test_batch=25

        for e in range(epochs):
            total_cost=0
            for ibatch in range(total_batch):
                radiomics_train,labels_train=sess.run(next_patch)
                radiomics_train2=np.reshape(radiomics_train,[12,104])
                labels_train2=np.reshape(labels_train,[1])
                labels_train2=1-labels_train2

                batch_cost,_=sess.run([cost, opt], feed_dict={inputs_: radiomics_train2,
                                                              labels_: labels_train2,learning_rate:lr})
                total_cost=total_cost+batch_cost
                
            print("Epoch: {}/{}".format(e+1, epochs))
            print(total_cost)
            predictresult=[]
            originalresult=[]
            possibility=[]
        for ibatch2 in range(test_batch):
            radiomics_test,labels_test=sess.run(next_patch2)
            radiomics_test2=np.reshape(radiomics_test,[12,104])
            labels_test2=np.reshape(labels_test,[1])
            labels_test2=1-labels_test2
            hypothesis_,Y_,alphas_=sess.run([hypothesis,Y,alphas], feed_dict={inputs_: radiomics_test2,
                                                                              labels_: labels_test2,learning_rate:lr})
        
            if hypothesis_[0,1]>0.5:
                predictresult.append(1)
            else:
                predictresult.append(0)
                
#        predictresult.append(np.argmax(hypothesis_))
            originalresult.append(np.argmax(Y_))
            possibility.append(hypothesis_[0,1])

            
        
        confusion_mat = confusion_matrix(originalresult, predictresult)
        precision= accuracy_score(originalresult, predictresult)
        fpr, tpr,tresholds = roc_curve(originalresult, possibility)
        recall=recall_score(originalresult, predictresult)
        auc=roc_auc_score(originalresult, possibility)
        PPV=confusion_mat[1,1]/(confusion_mat[0,1]+confusion_mat[1,1])
        NPV=confusion_mat[0,0]/(confusion_mat[0,0]+confusion_mat[1,0])
        AUC_final.append(auc)
        
        precision_final.append(precision)
        recall_final.append(recall)
        PPV_final.append(PPV)
        NPV_final.append(NPV)
        print("AUC:{} ".format(AUC_final[0]))
        print("Accuracy:{}".format(precision_final[0]))
        print("Recall:{}".format(recall_final[0]))
        print("NPV:{}".format(NPV_final[0]))
        print("PPV:{}".format(PPV_final[0]))

sess.close() 
