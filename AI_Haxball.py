import numpy as np
from PIL import ImageGrab, Image
import cv2
import time
import pyautogui
import random
import math
from mss import mss


player_center,player_radius, player_speed  = [],0,0.0
ball_center,ball_radius,ball_speed  = [],0,0.0
delta_time = 1

def show_image(image):
    cv2.imshow(str(random.randint(1,101)), image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        
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
        cv2.putText(image, "Player Center: {},{}".format(center[0],center[1]), (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
        player_center.append(center)
        player_radius = radius
    except:
        pass

def get_ball(image):
    try:
        f_img = color_filter(image,[255,255,255],True)
        f_img = cv2.cvtColor(f_img, cv2.COLOR_BGR2GRAY)
        im2 ,contours,hierarchy = cv2.findContours(f_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c)>300:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                (x,y),radius = cv2.minEnclosingCircle(c)
                center = (int(x),int(y))
                radius = int(radius)
                cv2.circle(image, center, radius, (255, 0, 0), 2)
                cv2.putText(image, "Ball Center: {},{}".format(center[0],center[1]), (200, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
                ball_center.append(center)
                ball_radius = radius               
    except:
        pass


 
    
    
    
    

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

        cv2.rectangle(image, (x1,y1), (x2,y2),[255,255,0],3)
    

    except:
        pass

def draw_circles(image,circles):
    try:
        print(len(circles[0]))
        for circle in circles[0]:
            x,y,r = circle
            cv2.circle(image, (x,y),r,(255,0,255),4)
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


    

    
    
    
def process_image(image):
##    pImg = color_filter(original_image,[255,0,0])
##    pImgGray = cv2.cvtColor(pImg, cv2.COLOR_BGR2GRAY)
##    pImgCanny = cv2.Canny(pImgGray, threshold1 = 200, threshold2 =300)
##    pImgBlur = cv2.GaussianBlur(pImgCanny, (5,5),0)
##    #edges
####    lines = cv2.HoughLinesP(processed_image_blur, 1,np.pi/180, 180, np.array([]), 600, 30)
##    circles = cv2.HoughCircles(pImgGray,cv2.HOUGH_GRADIENT, 1.2,5,
##                            param1=60,param2=30,minRadius=20,maxRadius=40 )
##
####    draw_lines(original_image, lines)
##    draw_circles(original_image,circles)
##    return pImg
   
    get_player(image)
    get_ball(image)
    if len(ball_center)>5:
        ball_speed = find_dist(ball_center[-1],ball_center[-2])/delta_time
        cv2.putText(image, "Ball Speed: {}".format(ball_speed), (200, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
    if len(player_center)>5:
        player_speed = find_dist(player_center[-1],player_center[-2])/delta_time
        cv2.putText(image, "Player Speed: {}".format(player_speed), (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)


    
def screen_record():
    sct = mss()
    
##    for i in range(1):
    while 1:
        last_time = time.time()
        monitor = {'top': 230, 'left': 30, 'width': 820, 'height': 400}

        
        img = np.array(sct.grab(monitor))
 
        
        processed_img = process_image(img)
        img = cv2.resize(img, (int(monitor['width']/3),int(monitor['height']/3)))
        delta_time = time.time()-last_time
        
        print('{} FPS'.format(1/delta_time))
        last_time = time.time()
        cv2.imshow('image', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

screen_record()
