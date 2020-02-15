# %%
'''
## 作業

1. 如何使用已經訓練好的模型？
2. 依照 https://github.com/qqwweee/keras-yolo3 的程式碼，請敘述，訓練模型時，資料集的格式是什麼？具體一點的說，要提供什麼格式的文件來描述資料集的圖片以及 bboxes 的信息呢？




'''

# %%
#%tensorflow_version 1.x # 確保 colob 中使用的 tensorflow 是 1.x 版本而不是 tensorflow 2
import tensorflow as tf
print(tf.__version__)

# %%
#pip install keras==2.2.4 # 需要安裝 keras 2.2.4 的版本

# %%
#from google.colab import drive
#drive.mount('/content/gdrive') # 將 google drive 掛載在 colob，
## 下載基於 keras 的 yolov3 程式碼
#%cd 'gdrive/My Drive'
## !git clone https://github.com/qqwweee/keras-yolo3 # 如果之前已經下載過就可以註解掉
#%cd keras-yolo3

# %%
from PIL import Image
image = Image.open('dog.jpg')

# %%
'''
1.如何使用已經訓練好的模型？

如果你理解了這包程式碼，其實就會知道可以直接從 yolo.py 從 include create YOLO 的 class，然後提供我們訓練好的模型檔案以及描述類別的文件就可以啦
'''

# %%
from yolo import YOLO
yolo_model = YOLO(model_path='logs/000/trained_weights_final.h5', classes_path="model_data/voc_classes.txt")
r_image = yolo_model.detect_image(image)

# %%
print (r_image)

# %%
'''
這個模型用了 pretrained 的權重以及 100 張圖片來 finetune，所以結果沒有很好，在這裡只是示範怎麼使用哦
'''

# %%
'''
2.請敘述，訓練模型時，資料集的格式是什麼？

這個問題背後的動機是希望確保你理解在訓練模型的時候我們需要把資料轉換成這份 YOLO 訓練程式碼“讀得懂”的格式。其實很簡單，只要把範例中 `convert_annotation` 這個函數給看懂就可以了
'''

# %%
with open("2007_train.txt", "r") as f:
  d = f.readlines()
print(d[:10])

# %%
'''
這是檔案的每一行對應的是一張圖片的路徑以及該圖片中物件的坐標及類別信息。首先是圖片路徑，然後以空白鍵區隔每個物件的信息，物件訊息的順序是包圍框的左上角 x,y，右下角 x, y 以及類別 index。
'''

# %%
