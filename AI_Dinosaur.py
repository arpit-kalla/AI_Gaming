import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui

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
    processed_image = cv2.Canny(processed_image, threshold1 = 200, threshold2 =300)
    processed_image = cv2.GaussianBlur(processed_image, (5,5),0)
    #edges
    lines = cv2.HoughLinesP(processed_image, 1,np.pi/180, 180, np.array([]), 50, 1)
    draw_lines(processed_image, lines)
    return processed_image

def screen_record(): 
    last_time = time.time()
    while(True):
        # 800x600 windowed mode
        printscreen =  np.array(ImageGrab.grab(bbox=(800,200,2050,500)))
        new_screen = process_image(printscreen)
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
##        cv2.imshow('Canny Edge', new_screen)
####        cv2.imshow('window',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
##        if cv2.waitKey(25) & 0xFF == ord('q'):
##            cv2.destroyAllWindows()
##            break

screen_record()
