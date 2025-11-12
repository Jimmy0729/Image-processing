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
img3 = cv2.imread('T3-test.jpg', 0)


'''
觀察圖片可以看到圖片有經過模糊化，可能是高斯模糊。
首先用反濾波嘗試，但發現原始圖片應該是有雜訊的，所以改用維那濾波
嘗試。
'''

# 1. 建立 PSF
psf = cv2.getGaussianKernel(11, 2)
psf = psf @ psf.T
psf = psf / psf.sum()

# PSF 對齊
psf_padded = np.zeros(img3.shape)
kh, kw = psf.shape
psf_padded[:kh, :kw] = psf
psf_padded = np.roll(psf_padded, -kh//2, axis=0)
psf_padded = np.roll(psf_padded, -kw//2, axis=1)

# 3. Wiener 濾波
G = np.fft.fft2(img3)
H = np.fft.fft2(psf_padded)

K = 0.035  
H_conj = np.conj(H)
H_abs_sq = np.abs(H)**2
F_hat = (H_conj / (H_abs_sq + K)) * G
restored = np.real(np.fft.ifft2(F_hat))
restored = np.clip(restored, 0, 255).astype(np.uint8)

# === 保存圖片 ===
# cv2.imwrite('restored_result.jpg', restored)

cv2.imshow('Inverse Filter Result', restored)
cv2.waitKey(0)









# --- 像素直方圖檢視
plt.figure(figsize=(6,4))
plt.title("Histogram intensity")

histogram(restored , color='gray')

plt.legend()
plt.show()