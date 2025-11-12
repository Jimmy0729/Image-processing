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
img5 = cv2.imread('T5-test.jpg', 0)


'''
觀察圖片可以看到圖片有經過模糊化，但不知道是哪種模糊方法。參考T3
的做法，先用維納濾波嘗試並進行 fine-tuning，當K太小時會出現振鈴，
所以最後將K設為0.05減少振鈴的出現。
'''

# 1. 建立 PSF
psf = cv2.getGaussianKernel(21, 3)
psf = psf @ psf.T
psf = psf / psf.sum()

# PSF 對齊
psf_padded = np.zeros(img5.shape)
kh, kw = psf.shape
psf_padded[:kh, :kw] = psf
psf_padded = np.roll(psf_padded, -kh//2, axis=0)
psf_padded = np.roll(psf_padded, -kw//2, axis=1)

# 3. Wiener 濾波
G = np.fft.fft2(img5)
H = np.fft.fft2(psf_padded)

K = 0.05
H_conj = np.conj(H)
H_abs_sq = np.abs(H)**2
F_hat = (H_conj / (H_abs_sq + K)) * G
restored = np.real(np.fft.ifft2(F_hat))
restored = np.clip(restored, 0, 255).astype(np.uint8)


# === 保存圖片 ===
cv2.imwrite('restored_result.jpg', restored)

cv2.imshow('Inverse Filter Result', restored)
cv2.waitKey(0)






# --- 像素直方圖檢視
# plt.figure(figsize=(6,4))
# plt.title("Histogram intensity")

# histogram(restored , color='gray')

# plt.legend()
# plt.show()