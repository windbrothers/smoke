# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 11:39:49 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 16:01:13 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 18:06:20 2020

@author: Administrator
"""
from keras.models import Sequential
#from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
#from keras.initializers import TruncatedNormal
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
import keras
#import tensorflow as tf
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
#import utils_paths
import matplotlib.pyplot as plt
import numpy as np
#import argparse
import random
import pickle
import cv2
from keras import optimizers
from keras import applications
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
 
 
def list_images(basePath, contains=None):
    # 返回有效的图片路径数据集
    return list_files(basePath, validExts=image_types, contains=contains)

def list_files(basePath, validExts=None, contains=None):
    # 遍历图片数据目录，生成每张图片的路径
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # 循环遍历当前目录中的文件名
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue
 
            # 通过确定.的位置，从而确定当前文件的文件扩展名
            ext = filename[filename.rfind("."):].lower()
 
            # 检查文件是否为图像，是否应进行处理
            if validExts is None or ext.endswith(validExts):
                # 构造图像路径
                imagePath = os.path.join(rootDir, filename)
                yield imagePath

def loadfile():
    # 读取数据和标签
    print("------开始读取数据-----")
    data = []
    labels = []
    # 拿到图像数据路径，方便后续读取
    imagePaths = sorted(list(list_images('./dataset/set10')))
    random.seed(42)
    random.shuffle(imagePaths)
    #print(imagePaths)
    # 遍历读取数据
    for imagePath in imagePaths:
        # 读取图像数据
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (64, 64))
        data.append(image)
        # 读取标签
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    # 对图像数据做scale操作
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    # 数据集切分
    (trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.20, random_state=42)
    #print(trainX.shape,testX.shape,trainY.shape,testY.shape)
    # 转换标签为one-hot encoding格式
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY) #该放多针对多分类有效
    testY = lb.transform(testY)
    #
    trainY = keras.utils.to_categorical(trainY,2)
    testY = keras.utils.to_categorical(testY,2)
    return trainX, testX, trainY, testY,lb
class SimpleVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
#        model = Sequential()
        inputShape = (height, width, depth)
#        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
#            chanDim = 1
#        model.add(Conv2D(64, (3, 3), padding="same",
#            input_shape=inputShape))

        base_model = applications.ResNet50(weights="imagenet", include_top=False,
                                        input_shape=inputShape)  # 预训练的VGG16网络，替换掉顶部网络
        print('base_model',base_model.summary())
        
#        for layer in base_model.layers[:15]: layer.trainable = False  # 冻结预训练网络前15层
        
        top_model = Sequential()  # 自定义顶层网络
        top_model.add(Flatten(input_shape=base_model.output_shape[1:]))  # 将预训练网络展平
        top_model.add(Dense(256, activation='relu'))  # 全连接层，输入像素256
        top_model.add(Dropout(0.5))  # Dropout概率0.5
        top_model.add(Dense(classes, activation='softmax'))  # 输出层，二分类
        print('top_model',top_model.summary())
        model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
        return model
if __name__ == '__main__':
    trainX, testX, trainY, testY,lb=loadfile()
    print(trainX.shape,testX.shape,trainY.shape,testY.shape,lb)
#    np.save('Train-img_data',trainX)
#    np.save('Train-img_label',trainY)
#    np.save('Test-img_data',testX)
#    np.save('Test-img_label',testY)
#    print('ok')
#    trainX = np.load('Train-img_data.npy')
#    testX=np.load('Train-img_label.npy')
#    trainY=np.load('Test-img_data.npy')
#    testY=np.load('Test-img_label.npy')
#    print(testY)
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")
## 建立卷积神经网络
    model = SimpleVGGNet.build(width=64, height=64, depth=3,classes=len(lb.classes_))
#    # 设置初始化超参数
    INIT_LR = 0.01
    EPOCHS = 50
    BS = 32
   # 损失函数，编译模型
    print("------准备训练网络------")
    opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
    model.summary()#显示模型
#    # 训练网络模型
#    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
#        validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
#        epochs=EPOCHS)
#    """
    H = model.fit(trainX, trainY, validation_data=(testX, testY),
        epochs=EPOCHS, batch_size=32)
#    """

#    # 测试
    print("------测试网络renset------")
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1),
        predictions.argmax(axis=1), target_names=lb.classes_))
#    
#    # 绘制结果曲线
    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.plot(N, H.history["accuracy"], label="train_acc")
    plt.plot(N, H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig('./output/renset.png')
    #
    # 保存模型
    print("------正在保存模型------")
    model.save('./output/renset.model')
    f = open('./output/renset.pickle', "wb")
    f.write(pickle.dumps(lb))
    f.close()