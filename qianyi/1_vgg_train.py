# 导入所需模块
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.initializers import TruncatedNormal
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
import keras
import tensorflow as tf

    

 
 




#print(trainY)
#print(testY)
#print(len(lb.classes_))
#print(testY)

# 数据增强处理
