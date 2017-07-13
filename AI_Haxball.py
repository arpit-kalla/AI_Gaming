import numpy as np
from PIL import ImageGrab, Image
import cv2
import time
import pyautogui
from mss import mss




##for i in range(4)[::-1]:
##    print(i+1)
##    time.sleep(1)
##
##print('space')
##pyautogui.keyDown('space')

def get_slope_bias(coords):
    x1 =coords[0]
    y1 =coords[1]
    x2 =coords[2]
    y2 =coords[3]

    isVertical = False
    if not -(1e-4)< x2-x1 < 1e-4 :
        m = (y2-y1)/(x2-x1)
        b = y1-m*x1
    else:
        isVertical = True
        m = 0
        b = x1
    return m,b,isVertical

def draw_lines(image, lines):
    try:
        horizontal = []
        vertical = []
        for line in lines:
            coords = line[0]
##            cv2.line(image, (coords[0],coords[1]),(coords[2],coords[3]), [255,0,0],3)
            m,b,isVertical = get_slope_bias(coords)
            if isVertical:
                vertical.append(b)
            else:
                horizontal.append(b)
        x1,y1,x2,y2 = find_field(vertical,horizontal)

        cv2.rectangle(image, (x1,y1), (x2,y2),[0,255,0],cv2.FILLED)
    

    except:
        pass
    
def find_field(vertical,horizontal):
    vertical = sorted(vertical)
    horizontal = sorted(horizontal)

    vLine  =[]
    hLine = []

    partition = 0
    for i in range(len(vertical)-1):
        if abs(vertical[i+1]-vertical[i])>40:
            vLine.append(int(sum(vertical[:i+1])/len(vertical[:i+1])))
            partition = i+1
    vLine.append(int(sum(vertical[partition:])/len(vertical[partition:])))

    for i in range(len(horizontal)-1):
        if abs(horizontal[i+1]-horizontal[i])>60:
            hLine.append(int(sum(horizontal[:i+1])/len(horizontal[:i+1])))
            partition = i+1
    hLine.append(int(sum(horizontal[partition:])/len(horizontal[partition:])))

    return vLine[0],hLine[0],vLine[-1],hLine[-1]

    
    
def process_image(original_image):
    processed_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.Canny(processed_image, threshold1 = 200, threshold2 =300)
    processed_image_blur = cv2.GaussianBlur(processed_image, (5,5),0)
    #edges
    lines = cv2.HoughLinesP(processed_image_blur, 1,np.pi/180, 180, np.array([]), 600, 30)

    draw_lines(original_image, lines)


    return processed_image

def screen_record():
    sct = mss()
##    for i in range(1):
    while 1:
        last_time = time.time()
        monitor = {'top': 230, 'left': 30, 'width': 820, 'height': 400}        
        img = np.array(sct.grab(monitor))
        processed_img = process_image(img)
        img_resize = cv2.resize(img, (int(monitor['width']/4),int(monitor['height']/4)))

        print('{} FPS'.format(1/(time.time()-last_time)))
        last_time = time.time()
        

        cv2.imshow('image', img_resize)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

screen_record()
