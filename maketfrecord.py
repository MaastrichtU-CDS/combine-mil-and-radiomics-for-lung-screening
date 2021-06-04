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

#类别
#tfrecords格式文件名
#writer1= tf.python_io.TFRecordWriter("A:/Dataset/NSCLC Radiogenomics files/LIDC-IDRI/Encoder-decoder Network/TFrecords/100 Epochs/MILTrain12.tfrecords") 
#writer2= tf.python_io.TFRecordWriter("A:/Dataset/NSCLC Radiogenomics files/LIDC-IDRI/Encoder-decoder Network/TFrecords/100 Epochs/MILTest12.tfrecords")
#datafile='A:/Dataset/NSCLC Radiogenomics files/LIDC-IDRI/Encoder-decoder Network/ExcelResults/normalizedsummary.xlsx'
#datafile2='A:/Dataset/NSCLC Radiogenomics files/LIDC-IDRI/Encoder-decoder Network/ExcelResults/simulationNegativeSummary.xlsx'
datafile='A:/Dataset/FileforLIDC/Summaryexcel/normalizedsummary.xlsx'
datafile2='A:/Dataset/FileforLIDC/Summaryexcel/simulationNegativeSummary.xlsx'
writer1= tf.python_io.TFRecordWriter("A:/Dataset/FileforLIDC/Summaryexcel/sensitivity analysis/SA40Train10.tfrecords") 
writer2= tf.python_io.TFRecordWriter("A:/Dataset/FileforLIDC/Summaryexcel/sensitivity analysis/SA40Test10.tfrecords")
#datafile2='A:/Datas
#line=datafile.readline()
count=0
count2=0
index=list(range(109))
np.random.shuffle(index)
#index2=list(range(100,110))
#index=index+index2
book = xlrd.open_workbook(datafile)#得到Excel文件的book对象，实例化对象
sheet0 = book.sheet_by_index(0) # 通过sheet索引获得sheet对象
book1 = xlrd.open_workbook(datafile2)#得到Excel文件的book对象，实例化对象
sheet1 = book1.sheet_by_index(0) # 通过sheet索引获得sheet对象
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
            #value=[index]决定了图片数据的类型label
        'RadiomicFeatures': tf.train.Feature(bytes_list=tf.train.BytesList(value=[roiradiomic_towrite])),
        'Label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_towrite]))
        })) #example对象对label和image数据进行封装 
    if patient in index[0:84]:
        count1=count1+1
        writer1.write(example.SerializeToString())  #序列化为字符串
    else:
        count2=count2+1
        writer2.write(example.SerializeToString())  #序列化为字符串
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
    #list(chain.from_iterable(roiradiomic))
    #print(label)
    radiomic_array=np.array(roiradiomic,dtype=np.float64)
    label_array=np.array(label)

    label_array2=np.array(label_array)
    roiradiomic_towrite=radiomic_array.tobytes()
    label_towrite=label_array2.tobytes()

    example = tf.train.Example(features=tf.train.Features(feature={
            #value=[index]决定了图片数据的类型label
        'RadiomicFeatures': tf.train.Feature(bytes_list=tf.train.BytesList(value=[roiradiomic_towrite])),
        'Label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_towrite]))
        })) #example对象对label和image数据进行封装 
    count1=count1+1
    writer1.write(example.SerializeToString())  #序列化为字符串
writer1.close()
writer2.close()