# %%
'''
## 『本次練習內容』
#### 搭建 Conv2D-BN-Activation層

'''

# %%
'''
## 『本次練習目的』
  #### 了解如何搭建CNN基礎架構，Conv2D-BN-Activation
'''

# %%
from tensorflow.keras.models import Sequential  #用來啟動 NN
from tensorflow.keras.layers import Conv2D  # Convolution Operation
from tensorflow.keras.layers import MaxPooling2D # Pooling
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense # Fully Connected Networks
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation

# %%
'''
## 依照指示建立模型
'''

# %%
input_shape = (32, 32, 3)

model = Sequential()

##  Conv2D-BN-Activation('sigmoid')

#BatchNormalization主要參數：
#momentum: Momentum for the moving mean and the moving variance.
#epsilon: Small float added to variance to avoid dividing by zero.

model.add(Conv2D(32, (3,3), strides=(1,1), padding="same", input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))


##、 Conv2D-BN-Activation('relu')
model.add(Conv2D(32, (3,3), strides=(1,1), padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.summary()

# %%
