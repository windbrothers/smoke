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


# 加载测试数据并进行相同预处理操作
image = cv2.imread('./cs_image/smoke1.jpg')
output = image.copy()
image = cv2.resize(image, (64, 64))

# scale图像数据
image = image.astype("float") / 255.0

# 对图像进行拉平操作
image = image.reshape((1, image.shape[0], image.shape[1],image.shape[2]))

# 读取模型和标签
print("------读取模型和标签------")
model = load_model('./output/qianyi2_VGG16.model')
lb = pickle.loads(open('./output/qianyi2_VGG16.pickle', "rb").read())

# 预测
preds = model.predict(image)

# 得到预测结果以及其对应的标签
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]

# 在图像中把结果画出来aaa
text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
print(text)
# 绘图
cv2.imshow("Image", output)
cv2.waitKey(0)