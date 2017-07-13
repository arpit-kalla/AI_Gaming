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

def draw_lines(image, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(image, (coords[0],coords[1]),(coords[2],coords[3]), [255,0,0],3)
    except:
        pass
    
def process_image(original_image):
    processed_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_image_canny = cv2.Canny(processed_image, threshold1 = 200, threshold2 =300)
    processed_image_canny = cv2.GaussianBlur(processed_image_canny, (5,5),0)
    #edges
    lines = cv2.HoughLinesP(processed_image_canny, 1,np.pi/180, 180, np.array([]), 50, 1)
    draw_lines(original_image, lines)
    return processed_image

def screen_record():
    sct = mss()
    while(True):
        last_time = time.time()
        monitor = {'top': 200, 'left': 0, 'width': 810, 'height': 420}        
        img = np.array(sct.grab(monitor))
        img_resize = cv2.resize(img, (405,210))
        processed_img = process_image(img)
        print('{} FPS'.format(1/(time.time()-last_time)))
        last_time = time.time()
        

        cv2.imshow('image', img_resize)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

screen_record()
