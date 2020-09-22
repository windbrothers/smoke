# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 14:22:25 2020

@author: Administrator
"""

# 导入所需工具包
from keras.models import load_model
#import argparse
import pickle
import cv2
# 用训练好的模型来预测新的样本
#from keras.preprocessing import image
import numpy as np

import os
 
def predict(model, img_path,lb, target_size):
    a=0
    b=0
    fileList = os.listdir(img_path)
    for fileName in fileList: 
#        print(fileName)
        img_path_name=img_path+fileName
        print(img_path_name)
#        img = cv2.imread(img_path_name)
        image = cv2.imread(img_path_name)
        output = image.copy()
        image = cv2.resize(image, (64, 64))
        
        # scale图像数据
        image = image.astype("float") / 255.0
        
        # 对图像进行拉平操作
        image = image.reshape((1, image.shape[0], image.shape[1],image.shape[2]))

        preds = model.predict(image)
        
        # 得到预测结果以及其对应的标签
        i = preds.argmax(axis=1)[0]
        label = lb.classes_[i]
        
        # 在图像中把结果画出来aaa
        text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
        cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
        print(text)
        # 绘图
#        cv2.imshow("Image", output)
#        cv2.waitKey(0)
        if(label=='smoke'):
            a=a+1
            path_S='./result/'+fileName
            print(path_S)
#            img = cv2.resize(img, (4000,3000))
            cv2.imwrite(path_S,output)
        else:
            b=b+1
            path_S='./result/no/'+fileName
            print(path_S)
#            img = cv2.resize(img, (4000,3000))
            cv2.imwrite(path_S,output) 
    print(b)
    return a,b
            
if __name__ == '__main__':
    model = load_model('./output/renset.h5')
    lb = pickle.loads(open('./output/renset.pickle', "rb").read())
    target_size = (64, 64)
    
#    img_path = './cs_image/smoke/'
    img_path = './1/'
    predict(model, img_path,lb, target_size)
#    print(a,b)
#    x=b/a
#    print(x)
#    print(res)

# 加载测试数据并进行相同预处理操作
