import cv2
import matplotlib.pyplot as plt
import numpy as np
def region_of_interest(image,corner = 'right'): 
    #get the resolution of the image
    height, width = image.shape
    map_perc = 0.85
    #set up the map extracting area
    map_height_limit = int(0.7*height)
    #set the cropping polygons
    if corner == 'right':
        map_width_limit = int(map_perc*width)
        area = [(map_width_limit, height),(map_width_limit, map_height_limit),(width, map_height_limit),(width, height),]
        crop_area = np.array([area], np.int32)
    if corner == 'left':
        map_width_limit_left = int((1-map_perc)*width)
        area = [(0, height),(0, map_height_limit),(map_width_limit_left, map_height_limit),(map_width_limit_left, height),]
        crop_area = np.array([area], np.int32)
    #set the background of the mask to 0
    mask = np.zeros_like(image)
    #get the mask done, the mask only allows minimap area to be further processed
    cv2.fillPoly(mask, crop_area, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def get_map_boundaries_right(x1,x2,y1,y2,a,c,b,d):
    xmin=min(x1,x2)
    xmax=max(x1,x2)
    ymin=min(y1,y2)
    ymax=max(y1,y2)
    if (abs(x1-x2)<5):#verdical boudary
        if (xmax<a):
            a=xmin
        elif (xmin>c):
            c=xmax
    if (abs(y1-y2)<5):#horizontal boudary
        if (ymax<b):
            b=ymin
        elif (ymin>d):
            d=ymax
    return [a,c,b,d]

def get_map_boundaries_left(x1,x2,y1,y2,a,c,b,d):
    xmin=min(x1,x2)
    xmax=max(x1,x2)
    ymin=min(y1,y2)
    ymax=max(y1,y2)
    if (abs(x1-x2)<5):#verdical boudary
        if (xmin>a):
            a=xmax
        elif (xmax<c):
            c=xmin
    if (abs(y1-y2)<5):#horizontal boudary
        if (ymax<b):
            b=ymin
        elif (ymin>d):
            d=ymax
    return [a,c,b,d]

def finding_minimap(image, lines, corner = 'right'):
    #get the hight,lenth of the image.
    y, x, c = image.shape
    #initialize the boudary coordinates(outside of the image)
    a = int(0.96*x)
    c = int(0.96*x)
    a_left = int(0.04*x)
    c_left = int(0.04*x)
    b = int(y*0.96)
    d = int(y*0.96)
    map = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            #get verdical/horizontal lines in the mini map
            ##right corner
            if corner == 'right':
                (a,c,b,d) = get_map_boundaries_right(x1,x2,y1,y2,a,c,b,d)
            ##left coner
            if corner == 'left':
                (a_left,c_left,b,d) = get_map_boundaries_left(x1,x2,y1,y2,a_left,c_left,b,d)

    if corner == 'right':
        map_co = [a,b,c,d]
    if corner == 'left':
        map_co = [a_left,b,c_left,d]
    return map_co

def display_map(image,a,b,c,d):
    map = np.zeros_like(image)
    #horizontal
    cv2.line(map, (a, b), (c, b), (0, 255, 0), 3)
    cv2.line(map, (a, d), (c, d), (0, 255, 0), 3)
    #verdical
    cv2.line(map, (a, b), (a, d), (0, 255, 0), 3)
    cv2.line(map, (c, b), (c, d), (0, 255, 0), 3)
    return map

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
    return line_image

def capture_map(cap, corner = 'right'):
    _, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #Thresholding to get map edges highlighted
    ret, o6 = cv2.threshold(gray_image, dark, thr, cv2.THRESH_BINARY_INV)
    map_region = region_of_interest(o6, corner)
    #set up hough transformation
    rho = 2
    theta = np.pi/180
    threshold = 100
    lines = cv2.HoughLinesP(map_region,rho, theta, threshold, np.array ([]), minLineLength=30, maxLineGap=6)
    map_info = finding_minimap(frame, lines, corner)
    return map_info

class miniMap():
    def __init__(self,a,b,c,d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    
    def get_centre(self):
        return [int((a+c)/2),int((b+d)/2)]
#'''
cap = cv2.VideoCapture("test.mp4")
dark = 4
thr = 18
max_val = 255
map_coord_stack = np.empty((21,4), dtype = int)
foundMap = False
map_corner = 'right'
k = 3
#mini map location initializing 
while (foundMap == False):
    for i in range(21):
        map_coord_stack[i] = capture_map(cap,map_corner)
    a = map_coord_stack[:,0]
    b = map_coord_stack[:,1]
    c = map_coord_stack[:,2]
    d = map_coord_stack[:,3]
    #print (map_coord_stack)
    #getting the initial map coordinates
    a = np.argmax(np.bincount(map_coord_stack[:,0]))
    b = np.argmax(np.bincount(map_coord_stack[:,1]))
    c = np.argmax(np.bincount(map_coord_stack[:,2]))
    d = np.argmax(np.bincount(map_coord_stack[:,3]))
    map = miniMap(a,b,c,d)
    box_centre = map.get_centre()
    print(box_centre)
    if (map_corner=='right')and(box_centre[0]>1180)and(box_centre[0]<1220)and (box_centre[1]>600):
        foudnMap = True
        break
    elif (map_corner=='left')and(box_centre[0]>70)and(box_centre[0]<100)and (box_centre[1]>600):
        foudnMap = True
        break
    else:
        map_corner = 'left' if (map_corner == 'right') else 'right'
        map_coord_stack = np.empty((21,4), dtype = int)

res = (1280, 720)
while(cap.isOpened()):
    print(map_corner)
    print(a,b,c,d)
    _, frame = cap.read()
    frame = cv2.resize(frame, res)
    map_info = display_map(frame, a,b,c,d)
    centre = [abs(a-c),abs(b-d)]
    combo_image = cv2.addWeighted(frame, 0.8, map_info, 1, 1)
    cv2.imshow("Image", combo_image)
    ##########
    #check if the map resized or moved to the other corner
    (a1,b1,c1,d1) = capture_map(cap,map_corner)
    new_centre = [abs(a1-c1),abs(c1-d1)]
    #if the new centre is significantly away from the old one(2 pixels horizontal and verdical)
    if (abs(centre[0]-new_centre[0]) > 2) and (abs(centre[1]-new_centre[1]) > 2):
        verd_hor_change_error = abs(centre[0]-new_centre[0])-abs(centre[1]-new_centre[1])
        #if the verdical and horizontal change is about 45 degrees
        #it means it is a legit resizing(instead of the influence of noise)
        if verd_hor_change_error <= 3:
            [a,b,c,d]=[a1,b1,c1,d1]
    
    frame_map = miniMap(a,b,c,d)
    box_centre = frame_map.get_centre()
    #check if the map location moved to the opposite corner
    if (map_corner=='right')and(not((box_centre[0]>1180)and(box_centre[0]<1220)and(box_centre[1]>600))):
        map_corner = 'left'
    elif (map_corner=='left')and(not((box_centre[0]>70)and(box_centre[0]<100)and(box_centre[1]>600))):
        map_corner = 'right'
    #press q to exit the video window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
#cv2.threshold(image, threshold, max_val, algorithm)


'''
image = cv2.imread("l5.png")
map_corner = 'left'
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
dark = 4
thr = 18
max_val = 255
#Thresholding to get map edges highlighted
ret, o5 = cv2.threshold(gray_image, thr, max_val, cv2.THRESH_TRUNC)
ret, o6 = cv2.threshold(o5, dark, thr, cv2.THRESH_BINARY_INV)
#Get the region of Interest
TRUNC_REGION = region_of_interest(o6, map_corner)
#set up hough transformation parameters
rho = 2
theta = np.pi/180
threshold = 80
#Hough Transformation
lines = cv2.HoughLinesP(TRUNC_REGION,rho, theta, threshold, np.array ([]), minLineLength=50, maxLineGap=6)
#Get the lines
(a,b,c,d) = finding_minimap(image, lines, map_corner)
map_info = display_map(image, a,b,c,d)
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