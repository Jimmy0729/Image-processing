import cv2
import numpy as np

#輸入圖片
img = cv2.imread('test_picture.jpg')
points = []

'''
1.首先建立遮罩圖檔->黑色背景+白色的圓。將遮罩圖檔與測試圖檔進行AND運算,白色部分
是我們所要的目標,也就是ROI。接著再將ROI輸出即為結果。

2.接著要取出roi內的平均RGB,為了方便運算,須先將遮罩圖轉
為單通道mask。接著從測試圖檔切下mask單通道值為255(白色的圓)的pixel座標,最後取平均即為平均
RGB值
'''
def circle():
    x_str, y_str= input('Please input coordinate and type:').split()
    x, y = int(x_str), int(y_str)
    mask = np.zeros(img.shape[:2], dtype = "uint8")
    cv2.circle(mask, (x, y), 200, 255, -1)
    roi = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("ROI", roi)
    cv2.waitKey(0)

    roi_pixels = img[mask == 255]
    mean_colors = roi_pixels.mean(axis = 0)
    b, g ,r = mean_colors
    print("R:", r, "G:", g, "B:", b)

        
"""
當多邊形畫好後,要來決定最後的ROI。首先要做一個全黑的mask。然後通過剛才選定的point list,
在mask上新稱實心的多邊形。最後再與原圖做bitwise_and 即可得出所求ROI
"""
    

def polylines():
    cv2.imshow('dogs', img)
    #window_name:綁定滑鼠事件的視窗名稱，必須是已經用 cv2.imshow() 打開的視窗, 綁定滑鼠事件的視窗名稱，必須是已經用 cv2.imshow() 打開的視窗
    #callback:當滑鼠事件發生時會被呼叫的函式
    cv2.setMouseCallback('dogs', mouse_event)
    key = cv2.waitKey(0);
    if key == 13:
        mask = np.zeros(img.shape[:2], dtype='uint8')
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], (255, 255, 255))
        roi = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow('roi', roi)
        cv2.waitKey(0)
        
    # cv2.destroyAllWindows()
    roi_pixels = img[mask ==255]
    b, g, r = roi_pixels.mean(axis=0)
    print("R", r, "G", g, "B", b)

"""
要畫多邊形,需要以左鍵點擊圖上的點,這些點會存在名為point的 list。並將list最後兩個點形成一條直線
最後再由這些點與直線形成多邊形。e.g:滑鼠左鍵用來決定多邊形的點，滑鼠右鍵會生成最終的多邊形
"""
def mouse_event(event, x, y, flag, parm):
    if event == 1:
        points.append((x,y))
        cv2.circle(img, (x, y), 2, (0, 0, 0), -1)
        if (len(points) > 1):
            cv2.line(img, points[-1], points[-2], (255, 255, 255), 2)
        cv2.imshow('dogs', img)
    
    elif event == 2:
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (255, 255, 255), 2)
        cv2.imshow('dogs', img)
    
        
        
   
    
    

'''
首先要輸入所需要畫的圖形種類(type)
當type == 'c'-> ROI 為圓形
當type == 'p'-> ROI 為多邊形
'''
def draw():
    while True:
        type = input('Please input type:')
        if type == "c":
            circle()
            break
    
        elif type == 'p':
            polylines()
            break

        else:
            print('please enter agnin')

draw()







