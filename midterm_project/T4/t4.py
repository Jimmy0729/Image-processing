import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram(f, color='black', label=None):
    hist = cv2.calcHist([f], [0], None, [256], [0,256])
    plt.plot(hist, color=color, label=label)
    plt.xlim([0, 256])
    plt.xlabel("Intensity")
    plt.ylabel("#Pixels")


#--- 1.讀取圖片
img4 = cv2.imread('T4-test.jpg', 0)

'''
在原始圖片中，有很強烈的雜訊，所以先採用中值濾波來處理。會採用較大
的濾波視窗。視窗半徑越大，抑制雜訊效果會比較好一點，尤其是像原圖這種有強烈雜訊的
圖片，但副作用就是細節會被模糊。
'''

img4 = cv2.medianBlur(img4, 7)
cv2.imshow('repair', img4)
cv2.waitKey(0)


# === 保存圖片 ===
# cv2.imwrite('restored_result.jpg', img4)



# --- 像素直方圖檢視
# plt.figure(figsize=(6,4))
# plt.title("Histogram intensity")

# histogram(img4 , color='gray')

# plt.legend()
# plt.show()