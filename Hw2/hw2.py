import numpy as np
import cv2 as cv2


def PSNR(f, g):
    MSE = 0.0
    nr, nc = f.shape
    for x in range(nr): 
        for y in range(nc):
            MSE += (float(f[x, y]) -float(g[x, y]))**2
    MSE/=(nr*nc)
    PSNR = 10*np.log10(255*255/MSE)
    return PSNR

# --- 1.載入與尺寸讀取
img = cv2.imread('cat2.jpeg', 0)
nr, nc = img.shape #length:2871, width:4301

# --- 2.獲取比例
try:
    scale = float(input("enter scale:"))
    if scale <= 0:
        print("比例需要大於0")
        exit()
except Exception as e:
    print("無效輸入:{e}")
    exit()

# --- 3.計算新比例
nr1, nc1 = int(nr*scale), int(nc*scale)


img2 = cv2.resize(img, (nc1, nr1), interpolation=cv2.INTER_AREA)#縮小圖像
img3 = cv2.resize(img2, (nc, nr), interpolation=cv2.INTER_LINEAR)#修復圖像


# --- 4.顯示結果
# cv2.imshow("Original Image", img)

cv2.imshow("Image Scaling", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- 5.PSNR計算
print('縮小圖像對比原始圖像', PSNR(img, img3))
