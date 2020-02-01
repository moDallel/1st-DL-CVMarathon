# %%
'''
## 『本次練習內容』
#### 使用Xception backbone做 Trnasfer Learning

'''

# %%
'''
## 『本次練習目的』
  #### 了解如何使用Transfer Learning
  #### 了解Transfer Learning的優點，可以觀察模型收斂速度
'''

# %%
'''
##### 可以自行嘗試多種架構
'''

# %
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K
import tensorflow.keras as keras
from tensorflow.keras.layers import Input

from keras.datasets import cifar10
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder


input_tensor = Input(shape=(32, 32, 3))
#include top 決定要不要加入 fully Connected Layer

'''Xception 架構'''

'''Resnet 50 架構'''
model=keras.applications.ResNet50(include_top=False, weights='imagenet',
                                    input_tensor=input_tensor,
                                    pooling=None, classes=10)
model.summary()


# %%
'''
## 添加層數
'''

# %%
x = model.output
x = tf.reshape(x, [-1, 32, 32, 2])
'''可以參考Cifar10實作章節'''
x = Conv2D(32, (3, 3), padding='same',
                 input_shape=x.shape[1:])(x)
x = Activation('relu')(x)
x = Conv2D(32, (3, 3))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

x = Conv2D(64, (3, 3), padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(64, (3, 3))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

x = Flatten()(x)
x = Dense(512)(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(units=10,activation='softmax')(x)
model = Model(inputs=model.input, outputs=predictions)
print('Model深度：', len(model.layers))


# %%
'''
## 鎖定特定幾層不要更新權重
'''

# %%
for layer in model.layers[:100]:
    layer.trainable = False
for layer in model.layers[100:]:
    layer.trainable = True

# %%
'''
## 準備 Cifar 10 資料
'''

# %%
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape) #(50000, 32, 32, 3)

## Normalize Data
def normalize(X_train,X_test):
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test


## Normalize Training and Testset
x_train, x_test = normalize(x_train, x_test)

## OneHot Label 由(None, 1)-(None, 10)
## ex. label=2,變成[0,0,1,0,0,0,0,0,0,0]
one_hot=OneHotEncoder()
y_train=one_hot.fit_transform(y_train).toarray()
y_test=one_hot.transform(y_test).toarray()

# %%
'''
## Training
'''

# %%
# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train,y_train,batch_size=32,epochs=100)

# %%
