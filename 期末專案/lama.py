import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from simple_lama_inpainting import SimpleLama


# ----- 模型不算太複雜，因此在CPU上跑即可 -----
_original_load = torch.jit.load
def force_cpu_load(*args, **kwargs):
    kwargs['map_location'] = torch.device('cpu') 
    return _original_load(*args, **kwargs)
torch.jit.load = force_cpu_load


# --- 全域變數 ---
drawing = False      # 是否正在塗抹
brush_size = 20      # 預設筆刷大小
mask = None          # 遮罩
img_display = None   # 顯示用的圖片 (包含紅色軌跡)
cur_mouse_pos = (-100, -100) # 紀錄滑鼠位置 (為了顯示筆刷游標)

def mouse_callback(event, x, y, flags, param):
    """處理滑鼠繪圖與游標移動"""
    global drawing, mask, img_display, cur_mouse_pos, brush_size

    # 更新滑鼠當前位置 (給主迴圈畫筆刷游標用)
    cur_mouse_pos = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        # 畫在遮罩上 (白色代表要修補)
        cv2.circle(mask, (x, y), brush_size, (255), -1)
        # 畫在顯示圖上 (紅色代表塗抹軌跡)
        cv2.circle(img_display, (x, y), brush_size, (0, 0, 255), -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(mask, (x, y), brush_size, (255), -1)
            cv2.circle(img_display, (x, y), brush_size, (0, 0, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def on_trackbar(val):
    """滑動條回調函數"""
    global brush_size
    # 限制最小筆刷為 1，避免 crash
    brush_size = max(1, val)

def main():
    global mask, img_display, cur_mouse_pos

    # 1. 讀取圖片
    img_path = "test_set/test3.jpg"  # 請確保此檔案存在
    img_original = cv2.imread(img_path)
    
    if img_original is None:
        print(f"錯誤：找不到圖片，請檢查路徑是否正確！")
        return

    
    h, w = img_original.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    img_display = img_original.copy()

    # 2. 設定視窗與滑動條
    # 關鍵：視窗名稱要變數化，確保前後一致
    WINDOW_NAME = "Inpainting Tool" 
    
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    
    # 建立滑動條：(名稱, 視窗名稱, 預設值, 最大值, 回調函數)
    cv2.createTrackbar("Brush Size", WINDOW_NAME, brush_size, 100, on_trackbar)

    print("=========================================")
    print(f"正在編輯：{img_path}")
    print("操作說明：")
    print("1. [滑動條] 調整筆刷大小")
    print("2. [左鍵] 塗抹要修補的區域")
    print("3. [S] 或 [Space] 開始修補")
    print("4. [R] 重置")
    print("5. [Q] 離開")
    print("=========================================")

    while True:
        # 複製一份當前的顯示圖，用來畫「懸浮筆刷」(不破壞原本的塗抹痕跡)
        img_temp = img_display.copy()
        
        # 畫出一個跟筆刷一樣大的空心圓圈，讓使用者知道筆刷範圍
        if cur_mouse_pos != (-100, -100):
            cv2.circle(img_temp, cur_mouse_pos, brush_size, (255, 255, 255), 1)

        cv2.imshow(WINDOW_NAME, img_temp)
        
        key = cv2.waitKey(10) & 0xFF # 稍微增加 delay 讓 UI 更順暢

        if key == ord('r'):
            mask = np.zeros((h, w), dtype=np.uint8)
            img_display = img_original.copy()
            print("遮罩已重置")

        elif key == ord('q') or key == 27:
            print("程式結束")
            cv2.destroyAllWindows()
            return

        elif key == ord('i'):
            if cv2.countNonZero(mask) == 0:
                print("尚未塗抹任何區域！")
                continue
            break

    cv2.destroyAllWindows()

    # 3. LaMa 推論部分
    try:
        print("正在運算中 (CPU 可能需要幾秒鐘)...")
        lama = SimpleLama(device='cpu')
        img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        result = lama(img_rgb, mask)
        
        if not isinstance(result, np.ndarray):
            result_rgb = np.array(result)
        else:
            result_rgb = result

        print("完成！")

        # 4. 顯示結果
        plt.figure(figsize=(18, 8))
        # --------第一張：原圖--------
        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.imshow(img_rgb)
        plt.axis('off')
        # --------第二張：遮罩圖--------
        plt.subplot(1, 3, 2)
        plt.title("Mask")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        # -------第三張：修復結果-------
        plt.subplot(1, 3, 3)
        plt.title("Inpainted Result")
        plt.imshow(result_rgb)
        plt.axis('off')

        plt.tight_layout() #自動調整三張圖的間距
        plt.show()

    except Exception as e:
        print(f"錯誤：{e}")

if __name__ == "__main__":
    main()