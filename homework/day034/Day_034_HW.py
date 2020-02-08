# %%
'''
## Day34.YOLO 細節理解 - 損失函數
今天的課程，我們講述了
* 損失函數是描述模型預測出來的結果和實際的差異的依據
* YOLO 損失函數的設計包含物件位置的定位與物件類別辨識
* YOLO損失函數透過超參數設定模型有不同的辨識能力


'''

# %%
'''
### 作業
仔細觀察，bbox 寬高計算損失方式和bbox中心計算損失方式有哪邊不一樣嗎? 為什麼要有不同的設計?
![title](loss function.png)


'''

# %%
'''
你的答案
Answer:
    (1) The loss of coordinate is Mean Square Error(MSE), but the loss of size
    (width/length) is assigned the square value before the MSE. Therefore,
    loss of size(width/length) try to renormalized the value by square value
    due to different meanings between coordinat and width/length. Difference
    of coordinate is a vector value between two points, but difference of
    width/length is the area difference value between two bbox.
'''
