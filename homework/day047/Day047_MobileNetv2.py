# %%
'''
# 作業
'''

# %%
'''
### 嘗試用 keras 的 DepthwiseConv2D 等 layers 實做 Inverted Residual Block.
   - depthwise's filter shape 爲 (3,3), padding = same
   - 不需要給 alpha, depth multiplier 參數
   - expansion 因子爲 6
'''

# %%
'''
##### 載入套件
'''

# %%
from tensorflow.keras.models import Model
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, BatchNormalization, ReLU, Add, Input

# %%
'''
##### 定義 Separable Convolution 函數
'''

def SeparableConv(input, output_dim=None):
    '''
    Args:
        input: input tensor
    Output:
        output: output tensor
    '''
    if output_dim is None:
       output_dim = int(input.shape[-1])
    # Depthwise Convolution
    x = DepthwiseConv2D((3,3), padding='same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Pointwise Convolution
    x = Conv2D(output_dim, (1,1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x

# %%
def InvertedRes(input, expansion):
    '''
    Args:
        input: input tensor
        expansion: expand filters size
    Output:
        output: output tensor
    '''
    # Expansion Layer
    x = Conv2D(expansion * 3, (1,1), padding='same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # SeparableConv layer
    x = SeparableConv(x, output_dim = 3)

    # Add Layer
    x = Add()([input, x])

    return x

# %%
'''
##### 建構模型
'''

# %%
input = Input((64, 64, 3))
output = InvertedRes(input, 6)
model = Model(inputs=input, outputs=output)
model.summary()

# %%
'''
更多相關連接參考: https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v2.py#L425
'''

# %%
