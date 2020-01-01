# %%
'''
## 『作業內容』
####   依照指示，透過調整Padding、Strides參數控制輸出Feature map大小

'''

# %%
'''
## 『目標』
####   了解輸出feature map尺寸變化原理
'''

# %%
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


##kernel size=(6,6)
##kernel數量：32
## Same padding、strides=(1,1)
classifier=Sequential()
inputs = Input(shape=(13,13,1))
x=Convolution2D(32, (3,3), input_shape=(13,13,1))(inputs)
model = Model(inputs=inputs, outputs=x)
model.summary()
## Same padding、strides=(2,2)
classifier=Sequential()
inputs = Input(shape=(13,13,1))
x=Convolution2D(32, (3,3), strides=(2,2))(inputs)
model = Model(inputs=inputs, outputs=x)
model.summary()
## Valid padding、strides=(1,1)
classifier=Sequential()
inputs = Input(shape=(13,13,1))
x=Convolution2D(32, (3,3), strides=(1,1), padding="valid")(inputs)
model = Model(inputs=inputs, outputs=x)
model.summary()
## Valid padding、strides=(2,2)
classifier=Sequential()
inputs = Input(shape=(13,13,1))
x=Convolution2D(32, (3,3), strides=(2,2), padding="valid")(inputs)
model = Model(inputs=inputs, outputs=x)
model.summary()
