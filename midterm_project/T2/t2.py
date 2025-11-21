import cv2
import numpy as np
import matplotlib.pyplot as plt

#---gamma矯正
def gamma_correction(f, gamma=1):
    table = ((np.arange(256) / 255.0) ** gamma * 255).astype(np.uint8)
    return cv2.LUT(f, table)

#---查看每張圖片的像素分佈情況

def histogram(f, color='black', label=None):
    hist = cv2.calcHist([f], [0], None, [256], [0,256])
    plt.plot(hist, color=color, label=label)
    plt.xlim([0, 256])
    plt.xlabel("Intensity")
    plt.ylabel("#Pixels")

#--- 1.讀取圖片
img2 = cv2.imread('T2-test.jpg', 0)

'''
觀察到圖片中有負片的效果，所以直接將 255剪掉像素值
及可以得到結果
'''
negative_img = 255 - img2
#--- T2


# === 保存圖片 ===
cv2.imwrite('restored_result.jpg', negative_img)

cv2.imshow("image_negative",negative_img)
cv2.waitKey(0)


# --- 像素直方圖檢視
# plt.figure(figsize=(6,4))
# plt.title("Histogram intensity")

# histogram(gamma_img , color='gray')

# plt.legend()
# plt.show()
