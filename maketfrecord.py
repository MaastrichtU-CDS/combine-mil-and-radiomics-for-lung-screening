# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 14:18:38 2020

@author: chenj
"""




import os 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image  
import matplotlib.pyplot as plt 
from itertools import chain
import numpy as np
import xlrd


datafile='A:/Dataset/FileforLIDC/Summaryexcel/normalizedsummary.xlsx'
datafile2='A:/Dataset/FileforLIDC/Summaryexcel/simulationNegativeSummary.xlsx'
writer1= tf.python_io.TFRecordWriter("A:/Dataset/FileforLIDC/Summaryexcel/sensitivity analysis/SA40Train10.tfrecords") 
writer2= tf.python_io.TFRecordWriter("A:/Dataset/FileforLIDC/Summaryexcel/sensitivity analysis/SA40Test10.tfrecords")

count=0
count2=0
index=list(range(109))
np.random.shuffle(index)

book = xlrd.open_workbook(datafile)
sheet0 = book.sheet_by_index(0) 
book1 = xlrd.open_workbook(datafile2)
sheet1 = book1.sheet_by_index(0) 
count=0
count1=0
count2=0
for patient in range(109):
    roiradiomic=[]
    label=[]
    for i in range(12):
        row_data =sheet0.row_values(count)
        for j in range(104):
            roiradiomic.append(float(row_data[j]))
        count=count+1
    label=int(row_data[-1])
    #print(label)
    #list(chain.from_iterable(roiradiomic))
    radiomic_array=np.array(roiradiomic,dtype=np.float64)
    label_array=np.array(label)
    label_array2=np.array(label_array)
    roiradiomic_towrite=radiomic_array.tobytes()
    label_towrite=label_array2.tobytes()

    example = tf.train.Example(features=tf.train.Features(feature={
           
        'RadiomicFeatures': tf.train.Feature(bytes_list=tf.train.BytesList(value=[roiradiomic_towrite])),
        'Label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_towrite]))
        })) 
    if patient in index[0:84]:
        count1=count1+1
        writer1.write(example.SerializeToString())  
    else:
        count2=count2+1
        writer2.write(example.SerializeToString())  
count3=0  
for patient in range(41):
    roiradiomic=[]
    label=[]
    for i in range(12):
        row_data =sheet1.row_values(count3)
        for j in range(104):
            roiradiomic.append(float(row_data[j]))
        count3=count3+1
    label=int(row_data[-1])

    radiomic_array=np.array(roiradiomic,dtype=np.float64)
    label_array=np.array(label)

    label_array2=np.array(label_array)
    roiradiomic_towrite=radiomic_array.tobytes()
    label_towrite=label_array2.tobytes()

    example = tf.train.Example(features=tf.train.Features(feature={
            
        'RadiomicFeatures': tf.train.Feature(bytes_list=tf.train.BytesList(value=[roiradiomic_towrite])),
        'Label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_towrite]))
        })) 
    count1=count1+1
    writer1.write(example.SerializeToString())  
writer1.close()
writer2.close()
