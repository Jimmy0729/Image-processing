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
img1 = cv2.imread('T1-test.jpg', 0)

'''
可以在圖中觀察到明顯的胡椒鹽雜訊，以及曝光高的問題，像素值偏右。
因此先用中值濾波將胡椒鹽濾除，再用gamma矯正改善圖像整體曝光高的問題。
'''
median = cv2.medianBlur(img1, 3)
gamma_correct = gamma_correction(median, gamma=1.4)


# === 保存圖片 ===
cv2.imwrite('restored_result.jpg', gamma_correct)

cv2.imshow('gamma correctoin', gamma_correct)
cv2.waitKey(0)
# cv2.destroyAllWindows()


# --- 像素直方圖檢視
# plt.figure(figsize=(6,4))
# plt.title("Histogram intensity")

# histogram(gamma_correct , color='gray')

# plt.legend()
# plt.show()