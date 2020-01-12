# %%
'''
## 『本次練習內容』
#### 運用這幾天所學觀念搭建一個CNN分類器
'''

# %%
'''
## 『本次練習目的』
  #### 熟悉CNN分類器搭建步驟與原理
  #### 學員們可以嘗試不同搭法，如使用不同的Maxpooling層，用GlobalAveragePooling取代Flatten等等
'''

# %%
from tensorflow.keras        import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.datasets import cifar10
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder


# %%
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape) #(50000, 32, 32, 3)

## Normalize Data
def normalize(X_train,X_test):
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test,mean,std


## Normalize Training and Testset
x_train, x_test,mean_train,std_train = normalize(x_train, x_test)

# %%
## OneHot Label 由(None, 1)-(None, 10)
## ex. label=2,變成[0,0,1,0,0,0,0,0,0,0]
one_hot=OneHotEncoder()
y_train=one_hot.fit_transform(y_train).toarray()
y_test=one_hot.transform(y_test).toarray()

# %%

classifier=Sequential()

#卷積組合
classifier.add(Convolution2D(32, (3,3), strides=(1,1), padding="same"))#32,3,3,input_shape=(32,32,3),activation='relu''
classifier.add(Activation("relu"))
classifier.add(BatchNormalization())
classifier.add(Convolution2D(32, (3,3), strides=(1,1), padding="same"))#32,3,3,input_shape=(32,32,3),activation='relu''
classifier.add(Activation("relu"))
classifier.add(BatchNormalization())

'''自己決定MaxPooling2D放在哪裡'''
classifier.add(MaxPooling2D(pool_size=(2,2)))

#卷積組合
classifier.add(Convolution2D(64, (3,3), strides=(1,1)))
classifier.add(BatchNormalization())
classifier.add(Activation("relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#flatten
classifier.add(Flatten())
#FC
classifier.add(Dense(units=512, activation="relu")) #output_dim=100,activation=relu

#輸出
classifier.add(Dense(units=10,activation='softmax'))

#超過兩個就要選categorical_crossentrophy
adam = optimizers.Adam(learning_rate = 0.0001, beta_1=0.9, beta_2=0.999, decay=1e-6)
classifier.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(x_train,y_train,batch_size=100,epochs=10)

# %%
'''
## 預測新圖片，輸入影像前處理要與訓練時相同
#### ((X-mean)/(std+1e-7) ):這裡的mean跟std是訓練集的
## 維度如下方示範
'''

# %%
input_example=(np.zeros(shape=(1,32,32,3))-mean_train)/(std_train+1e-7)
classifier.predict(input_example)
