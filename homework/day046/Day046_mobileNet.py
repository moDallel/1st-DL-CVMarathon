# %%
'''
# 作業
'''

# %%
'''
### 嘗試用 keras 的 DepthwiseConv2D 等 layers 實做 Separable Convolution.
   - depthwise's filter shape 爲 (3,3), padding = same
   - pointwise's filters size 爲 128
   - 不需要給 alpha, depth multiplier 參數
'''

# %%
'''
##### 載入套件
'''

# %%
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.layers import DepthwiseConv2D, Conv2D, BatchNormalization, ReLU, Input

# %%
'''
##### 定義 Separable Convolution 函數
'''

# %%
def SeparableConv(input):
    '''
    Args:
        input: input tensor
    Output:
        output: output tensor
    '''
    # Depthwise Convolution
    x = DepthwiseConv2D((3,3), padding='same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Pointwise Convolution
    x = Conv2D(128, (1,1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x

# %%
'''
##### 建構模型
'''

# %%
input = Input((64, 64, 3))
output = SeparableConv(input)
model = Model(inputs=input, outputs=output)
model.summary()

# %%
'''
更多相關連接參考: https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet.py#L364
'''

# %%
