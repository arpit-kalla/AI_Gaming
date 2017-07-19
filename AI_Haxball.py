import threading
import time
import cv2
import numpy as np
from mss import mss
import math
import pyautogui

player_center,player_radius, player_speed,player_acc  = [],0,[],0.0
ball_center,ball_radius,ball_speed,player_acc  = [],0,[],0.0
delta_time = 1

# Shows an Image in a new window
def show_image(image):
    cv2.imshow(str(random.randint(1,101)), image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

# Finds the distance between two points       
def find_dist(pos1,pos2):
    x1,y1 = pos1[0],pos1[1]
    x2,y2 = pos2[0],pos2[1]
    return math.sqrt((x2-x1)**2+(y2-y1)**2)

# Enter a RGB color and the funtion returns the HSV equivalent
def rgb2hsv(rgb):
    r,g,b = rgb
    
    r_prime = r/255
    g_prime = g/255
    b_prime = b/255

    Cmax = max(r_prime,g_prime,b_prime)
    Cmin = min(r_prime,g_prime,b_prime)
    delta = Cmax-Cmin

    if delta == 0:
        h = 0
    elif Cmax == r_prime:
        h = 60*(((g_prime-b_prime)/delta)%6)
    elif Cmax == g_prime:
        h = 60*(((b_prime-r_prime)/delta)+2)
    elif Cmax == b_prime:
        h = 60*(((r_prime-g_prime)/delta)+4)

    if Cmax == 0:
        s = 0
    else:
        s = delta/Cmax

    v = Cmax

    return h,int(s*255),int(v*255)


#Converts the coordinates to equivalent slope and intercept 
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

#Enter a image and color and it filters the objects of that color in that image
def color_filter(image,color,pure):
    hsvImg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_color = np.array(rgb2hsv(color))

    if pure == False:
        delta = np.array([10,100,100])
    else:
        delta = np.array([10,10,10])
    
    lower = hsv_color  - delta
    upper = hsv_color  + delta

    mask = cv2.inRange(hsvImg, lower, upper)
    res = cv2.bitwise_and(image,image, mask= mask)
    return res

        
# Get information of the Player
def get_player(image):
    try:
        f_img = color_filter(image,[255,0,0],False)
        f_img = cv2.cvtColor(f_img, cv2.COLOR_BGR2GRAY)
        im2 ,contours,hierarchy = cv2.findContours(f_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = contours[0]
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        (x,y),radius = cv2.minEnclosingCircle(c)
        center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(image, center, radius, (255, 0, 0), 2)
        
        player_center.append(center)
        player_radius = radius
    except:
        pass

# Get information of the Ball
def get_ball(image):
    try:
        f_img = color_filter(image,[255,255,255],True)
        f_img = cv2.cvtColor(f_img, cv2.COLOR_BGR2GRAY)
        im2 ,contours,hierarchy = cv2.findContours(f_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c)>60:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                (x,y),radius = cv2.minEnclosingCircle(c)
                center = (int(x),int(y))
                radius = int(radius)
                cv2.circle(image, center, radius, (255, 0, 0), 2)
                ball_center.append(center)
                ball_radius = radius               
    except:
        pass


# Prints text on the Image
def show_data(image):
    if len(ball_center)>1:
        cv2.putText(image, "Ball Center: {},{}".format(ball_center[-1][0],ball_center[-1][1]), (200, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
        ball_speed.append(find_dist(ball_center[-1],ball_center[-2])/delta_time)
        cv2.putText(image, "Ball Speed: {}".format(ball_speed[-1]), (200, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
    if len(player_center)>1:
        cv2.putText(image, "Player Center: {},{}".format(player_center[-1][0],player_center[-1][1]), (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
        player_speed.append(find_dist(player_center[-1],player_center[-2])/delta_time)
        cv2.putText(image, "Player Speed: {}".format(player_speed[-1]), (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)

    if len(player_speed)>1:
        player_acc = abs(player_speed[-1]-player_speed[-2])/delta_time
        cv2.putText(image, "Player Acc: {}".format(player_acc), (15, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
    if len(ball_speed)>1:
        ball_acc = abs(ball_speed[-1]-ball_speed[-2])/delta_time
        cv2.putText(image, "Ball Acc: {}".format(ball_acc), (200, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)

#All Processing of the image is done here       
def process_image(image):
    get_player(image)
    get_ball(image)
    show_data(image)

# Grab Screenshots of the Screen
def get_screen():
    sct = mss()
    while 1:
        last_time = time.time()
        monitor = {'top': 230, 'left': 30, 'width': 820, 'height': 400}
        img = np.array(sct.grab(monitor))
        img = cv2.resize(img, (int(monitor['width']/3),int(monitor['height']/3)))
        
        processed_img = process_image(img)

        delta_time = time.time()-last_time
        
##        print('{} FPS'.format(1/delta_time))
        last_time = time.time()
        cv2.rectangle(img, (10,10), (20,20), (255, 0, 0), 1)
        cv2.imshow('image', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def emulate():
    x,y,buffer = 276,132,10
    while True:
        if len(player_center)>0 and len(ball_center)>0:
            pX,pY,pR = player_center[-1][0], player_center[-1][1], player_radius
            bX,bY,bR = ball_center[-1][0],   ball_center[-1][1],   ball_radius
            print(pX,pY)
            if pX-x>buffer:
                print("a")
                pyautogui.keyDown('a')
                pyautogui.keyUp('a')
            elif x-pX>buffer:
                print("d")
                pyautogui.keyDown('d')
                pyautogui.keyUp('d')

            if pY-y>buffer:
                print("w")
                pyautogui.keyDown('w')
                pyautogui.keyUp('w')
                
            elif y-pY>buffer:
                print("s")
                pyautogui.keyDown('s')
                pyautogui.keyUp('s')
            
            

        

emulator = threading.Thread(target = emulate)
emulator.daemon = True
emulator.start()
get_screen()
