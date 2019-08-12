import cv2
import matplotlib.pyplot as plt
import numpy as np
def region_of_interest(image): 
    #get the resolution of the image
    height, width = image.shape
    map_perc = 0.85
    #set up the map extracting area
    map_height_limit = int(0.7*height)
    map_width_limit = int(map_perc*width)
    map_width_limit_left = int((1-map_perc)*width)
    rightmap_area = [(map_width_limit, height),(map_width_limit, map_height_limit),(width, map_height_limit),(width, height),]
    leftmap_area = [(0, height),(0, map_height_limit),(map_width_limit_left, map_height_limit),(map_width_limit_left, height),]
    #set the cropping polygons
    crop_area = np.array([rightmap_area,leftmap_area], np.int32)
    #set the background of the mask to 0
    mask = np.zeros_like(image)
    #get the mask done, the mask only allows minimap area to be further processed
    cv2.fillPoly(mask, crop_area, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def finding_minimap(image, lines, corner = 'right'):
    #get the hight,lenth of the image.
    y, x, c = image.shape
    #initialize the boudary coordinates(outside of the image)
    ver_boudary_a = int(0.96*x)
    ver_boudary_c = int(0.96*x)
    ver_boudary_a_left = int(0.04*x)
    ver_boudary_c_left = int(0.04*x)
    hor_boudary_b = int(y*0.96)
    hor_boudary_d = int(y*0.96)
    mapcentre_x = x
    mapcentre_y = y
    map = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            #get verdical/horizontal lines in the mini map
            xmin=min(x1,x2)
            xmax=max(x1,x2)
            ymin=min(y1,y2)
            ymax=max(y1,y2)
            #right corner
            if corner == 'right':
                if (abs(x1-x2)<3):#verdical boudary
                    if (xmax<ver_boudary_a):
                        ver_boudary_a=xmin
                    elif (xmin>ver_boudary_c):
                        ver_boudary_c=xmax
                if (abs(y1-y2)<3):#horizontal boudary
                    if (ymax<hor_boudary_b):
                        hor_boudary_b=ymin
                    elif (ymin>hor_boudary_d):
                        hor_boudary_d=ymax       
            #left coner
            if corner == 'left':
                if (abs(x1-x2)<3):#verdical boudary
                    if (xmin>ver_boudary_a_left):
                        ver_boudary_a_left=xmax
                    elif (xmax<ver_boudary_c):
                        ver_boudary_c_left=xmin
                if (abs(y1-y2)<3):#horizontal boudary
                    if (ymax<hor_boudary_b):
                        hor_boudary_b=ymin
                    if (ymin>hor_boudary_d):
                        hor_boudary_d=ymax

    #display the boudaries on the map
    if (corner == 'right') :
        #horizontal
        cv2.line(map, (ver_boudary_a, hor_boudary_b), (ver_boudary_c, hor_boudary_b), (0, 255, 0), 3)
        cv2.line(map, (ver_boudary_a, hor_boudary_d), (ver_boudary_c, hor_boudary_d), (0, 255, 0), 3)
        #verdical
        cv2.line(map, (ver_boudary_a, hor_boudary_b), (ver_boudary_a, hor_boudary_d), (0, 255, 0), 3)
        cv2.line(map, (ver_boudary_c, hor_boudary_b), (ver_boudary_c, hor_boudary_d), (0, 255, 0), 3)
    if (corner == 'left') :
        #horizontal
        cv2.line(map, (ver_boudary_a_left, hor_boudary_b), (ver_boudary_c_left, hor_boudary_b), (0, 255, 0), 3)
        cv2.line(map, (ver_boudary_a_left, hor_boudary_d), (ver_boudary_c_left, hor_boudary_d), (0, 255, 0), 3)
        #verdical
        cv2.line(map, (ver_boudary_a_left, hor_boudary_b), (ver_boudary_a_left, hor_boudary_d), (0, 255, 0), 3)
        cv2.line(map, (ver_boudary_c_left, hor_boudary_b), (ver_boudary_c_left, hor_boudary_d), (0, 255, 0), 3)
    
    #display the centre of the map
    mapcentre_x = int((ver_boudary_a+ver_boudary_c)/2) if corner == 'right' else int((ver_boudary_a_left+ver_boudary_c_left)/2)
    mapcentre_y = int((hor_boudary_b+hor_boudary_d)/2)
    centre=(mapcentre_x,mapcentre_y)
    print(centre)
    cv2.circle(map, centre, 15, (0,255,255), -2)
    return map

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
    return line_image

#'''
cap = cv2.VideoCapture("test2.mp4")
#cap = cv2.VideoCapture("./testImage/Youtube_gameplay.mp4")
thr = 64
max_val = 255

while(cap.isOpened()):
    _, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    ret, o5 = cv2.threshold(gray_image, thr, max_val, cv2.THRESH_TRUNC)
    #ret, o5 = cv2.threshold(gray_image, 16, 255, cv2.THRESH_TOZERO)
    ret, o6 = cv2.threshold(o5, 4, 64, cv2.THRESH_BINARY_INV)
    TRUNC_REGION = region_of_interest(o6)
    rho = 2
    theta = np.pi/180
    threshold = 120
    lines = cv2.HoughLinesP(TRUNC_REGION,rho, theta, threshold, np.array ([]), minLineLength=50, maxLineGap=0)
    #line_image = display_lines(frame, lines)
    map_info = finding_minimap(frame, lines, 'right')
    combo_image = cv2.addWeighted(frame, 0.8, map_info, 1, 1)
    cv2.imshow("Image", combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
#cv2.threshold(image, threshold, max_val, algorithm)

'''

image = cv2.imread("l2.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
dark = 4
thr = 18
max_val = 255
#Thresholding to get map edges highlighted
ret, o5 = cv2.threshold(gray_image, thr, max_val, cv2.THRESH_TRUNC)
ret, o6 = cv2.threshold(o5, dark, thr, cv2.THRESH_BINARY_INV)
#Get the region of Interest
TRUNC_REGION = region_of_interest(o6)
#set up hough transformation parameters
rho = 2
theta = np.pi/180
threshold = 80
#Hough Transformation
lines = cv2.HoughLinesP(TRUNC_REGION,rho, theta, threshold, np.array ([]), minLineLength=70, maxLineGap=6)
#Get the lines
map_info = finding_minimap(image, lines, 'left')
hough = display_lines(image, lines)
hough_image = cv2.addWeighted(image, 0.8, hough, 1, 1)
combo_image = cv2.addWeighted(image, 0.8, map_info, 1, 1)
#cv2.imshow("Image", combo_image)
#cv2.waitKey(0)
plt.figure()
plt.imshow(o5)

plt.figure()
plt.imshow(o6)

plt.figure()
plt.imshow(TRUNC_REGION)

plt.figure()
plt.imshow(hough_image)

plt.figure()
plt.imshow(combo_image)

plt.show()
'''