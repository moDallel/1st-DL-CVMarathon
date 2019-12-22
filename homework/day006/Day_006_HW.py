# %%
'''
# 作業

練習以旋轉變換 + 平移變換來實現仿射變換
> 旋轉 45 度 + 縮放 0.5 倍 + 平移 (x+100, y-50)
'''

# %%
import cv2
import time
import numpy as np

img = cv2.imread('../data/lena.png')

# %%
'''
## Affine Transformation - Case 2: any three point
'''

# %%
# 給定兩兩一對，共三對的點
# 這邊我們先用手動設定三對點，一般情況下會有點的資料或是透過介面手動標記三個點
rows, cols = img.shape[:2]
pt1 = np.array([[50,50], [300,100], [200,300]], dtype=np.float32)
pt2 = np.array([[80,80], [330,150], [300,300]], dtype=np.float32)

# 取得 affine 矩陣並做 affine 操作
M_rotate = cv2.getRotationMatrix2D((cols//2, rows//2), 45, 0.5)
M_shift  = np.array([[1, 0, 100],
                     [0, 1, -50]],dtype=np.float32)
import numpy as np
theta    = np.deg2rad(45)
#M_affine =
#img_affine = cv2.warpAffine(img, M_rotate, (cols, rows))
#img_affine = cv2.warpAffine(img_affine, M_shift, (cols, rows))
A        = np.array([
                      [pt1[0][0], pt1[0][1], 1, 0, 0, 0],
                      [0, 0, 0, pt1[0][0], pt1[0][1], 1],
                      [pt1[1][0], pt1[1][1], 1, 0, 0, 0],
                      [0, 0, 0, pt1[1][0], pt1[1][1], 1],
                      [pt1[2][0], pt1[2][1], 1, 0, 0, 0],
                      [0, 0, 0, pt1[2][0], pt1[2][1], 1],
                      ], dtype=np.float32)
import scipy as sp
from scipy import linalg
A_inv = linalg.inv(A)
M_vec = np.dot(A_inv, (pt2).flatten())
M_affine_all = np.row_stack((M_vec.reshape(-1,3), [0., 0., 1.]))
M_affine     = M_vec.reshape(-1,3)
img_affine = cv2.warpAffine(img, M_affine, (cols, rows))
# check M_affine
#for pi, ptt in enumerate(pt1):
#    print(np.dot(M_affine_all, [ptt[0], ptt[1], 1]))
# 在圖片上標記點
img_copy = img.copy()
for idx, pts in enumerate(pt1):
    pts = tuple(map(int, pts))
    cv2.circle(img_copy, pts, 3, (0, 255, 0), -1)
    cv2.putText(img_copy, str(idx), (pts[0]+5, pts[1]+5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

for idx, pts in enumerate(pt2):
    pts = tuple(map(int, pts))
    cv2.circle(img_affine, pts, 3, (0, 255, 0), -1)
    cv2.putText(img_affine, str(idx), (pts[0]+5, pts[1]+5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

# 組合 + 顯示圖片
img_show_affine = np.hstack((img_copy, img_affine))
while True:
    cv2.imshow('affine transformation', img_show_affine)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        break
