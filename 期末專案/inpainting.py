import cv2
import numpy as np
import sys
import os


# -----全域變數-----
img = None           # 原始影像
img_show = None      # 顯示用影像 (含紅色塗抹痕跡)
mask = None          # 遮罩
drawing = False      # 是否正在畫
brush_size = 10      # 筆刷大小
cur_mouse_pos = (-100, -100) # 滑鼠位置 (為了顯示白色圈圈)
WINDOW_NAME = "Inpainting Tool"


# -----滑鼠事件 callback-----
def on_mouse(event, x, y, flags, param):
    global drawing, img_show, mask, brush_size, cur_mouse_pos

    # 更新滑鼠位置 (為了畫懸浮游標)
    cur_mouse_pos = (x, y)

    # 按下左鍵 or 拖曳 -> 畫圖
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(mask, (x, y), brush_size, 255, -1)
        cv2.circle(img_show, (x, y), brush_size, (0, 0, 255), -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(mask, (x, y), brush_size, 255, -1)
            cv2.circle(img_show, (x, y), brush_size, (0, 0, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(mask, (x, y), brush_size, 255, -1)
        cv2.circle(img_show, (x, y), brush_size, (0, 0, 255), -1)


# -----調整筆刷大小-----
def on_trackbar(val):
    global brush_size
    brush_size = max(1, val) # 確保至少為 1


# -----主程式-----
def main():
    global img, img_show, mask, brush_size, cur_mouse_pos

    # 1. 讀取影像 (預設 test.jpg)
    folder_name = 'test_set'
    filename = sys.argv[1]
    img_path = os.path.join(folder_name, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f" 無法讀取影像：{img_path}")
        return

    # 備份與初始化
    img_original = img.copy()
    img_show = img.copy()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # 2. 建立視窗
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    
    # 建立滑動條 (直接用這個調大小)
    cv2.createTrackbar("Brush Size", WINDOW_NAME, brush_size, 100, on_trackbar)

    print("=========================================")
    print(f"正在編輯：{img_path}")
    print(" 操作說明：")
    print("   1. [滑動條] 調整筆刷大小")
    print("   2. [左鍵] 塗抹紅色區域")
    print("   3. [i] 開始修復 (Inpaint)")
    print("   4. [r] 重置")
    print("   5. [q] 離開")
    print("=========================================")

    while True:
        # 複製畫面用來畫「懸浮筆刷」 (不影響實際塗抹)
        img_temp = img_show.copy()
        
        # 顯示白色空心圓圈，可以讓使用者得知目前筆刷大小
        if cur_mouse_pos != (-100, -100):
            cv2.circle(img_temp, cur_mouse_pos, brush_size, (255, 255, 255), 1)

        cv2.imshow(WINDOW_NAME, img_temp)
        
        key = cv2.waitKey(10) & 0xFF

        # [i] 執行修復
        if key == ord('i'):
            print("修復中...")
            result = cv2.inpaint(img, mask, 10, cv2.INPAINT_NS)
            cv2.imshow("Result", result)
            cv2.imwrite(img_path+'_repair.jpg', result)
            print("完成！")

        # [r] 重置
        elif key == ord('r'):
            print("重置")
            img = img_original.copy()
            img_show = img.copy()
            mask[:] = 0

        # [q] 離開
        elif key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()