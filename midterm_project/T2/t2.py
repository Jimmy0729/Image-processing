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
因為原圖對比度過高，所以採用自適應性直方圖等化CLAHE，因為發現直接直方圖等化好像
有點過度增強，所以改採用CLAHE，最後再用gamma修正稍微讓整體變暗一點
'''
#--- T2
# img1 = cv2.equalizeHist(imgs[1])
# negative_image = img_negative(imgs[1])
# gamma_correct2 = gamma_correction(imgs[1], gamma=1.2)
clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(12,12))
clahe_img = clahe.apply(img2)

# gaussian = cv2.GaussianBlur(clahe_img,(3,3),0.8)

gamma_img = gamma_correction(clahe_img, gamma=1.2)

# === 保存圖片 ===
# cv2.imwrite('restored_result.jpg', gamma_img)

cv2.imshow("image_negative",gamma_img)
cv2.waitKey(0)


# --- 像素直方圖檢視
# plt.figure(figsize=(6,4))
# plt.title("Histogram intensity")

# histogram(gamma_img , color='gray')

# plt.legend()
# plt.show()
