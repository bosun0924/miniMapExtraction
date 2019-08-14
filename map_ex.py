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
        return [int((self.a+self.c)/2),int((self.b+self.d)/2)]

    def get_boudaries(sefl):
        return [self.a, self.b, self.c, self.d]
#'''

def initial_detecting(cap,dark,thr,max_val,map_corner = 'right',foundMap = False,map_coord_stack = np.empty((55,4), dtype = int), k = 0):
    print("##############initializing#############")
    while (foundMap==False):
        for i in range(55):
            map_coord_stack[i] = capture_map(cap,map_corner)
        '''
        a = map_coord_stack[:,0]
        b = map_coord_stack[:,1]
        c = map_coord_stack[:,2]
        d = map_coord_stack[:,3]
        '''
        #print (map_coord_stack)
        #getting the initial map coordinates
        a = np.argmax(np.bincount(map_coord_stack[:,0]))
        b = np.argmax(np.bincount(map_coord_stack[:,1]))
        c = np.argmax(np.bincount(map_coord_stack[:,2]))
        d = np.argmax(np.bincount(map_coord_stack[:,3]))
        minimap = miniMap(a,b,c,d)
        box_centre = minimap.get_centre()
        print(box_centre)
        if (map_corner=='right')and(box_centre[0]>1180)and(box_centre[0]<1220)and (box_centre[1]>600):
            foudnMap = True
            break
        elif (map_corner=='left')and(box_centre[0]>70)and(box_centre[0]<100)and (box_centre[1]>600):
            foudnMap = True
            break
        else:
            map_corner = 'left' if (map_corner == 'right') else 'right'
            map_coord_stack = np.empty((55,4), dtype = int)
    print (a,b,c,d)
    return [a,b,c,d,map_corner]

#initializing 
cap = cv2.VideoCapture("final_test.mp4")
dark = 4
thr = 18
max_val = 255
map_corner = 'right'
#Locate the minimap
(a,b,c,d,map_corner) = initial_detecting(cap,dark,thr,max_val)
print(a,b,c,d)
res = (1280, 720)
#Track the minimap size/location changes
while(cap.isOpened()):
    #print(map_corner)
    #print(a,b,c,d)
    _, frame = cap.read()
    map_area = cv2.resize(frame, res)
    #printing out result
    '''
    map_info = display_map(map_area, a,b,c,d)
    #Show the whold image with mini map in a box at the corner
    result_image = cv2.addWeighted(map_area, 0.8, map_info, 1, 1)
    cv2.imshow("Image", result_image)
    '''
    #minimap cropping box
    ver_l = min(a,c)
    ver_r = max(a,c)
    result_image = cv2.resize(map_area[b:d,ver_l:ver_r], (252,252))
    cv2.imshow("mini map", result_image)
    #'''
    frame_map = miniMap(a,b,c,d)
    box_centre = frame_map.get_centre()
    #print(box_centre)
    ##########
    #check if the map resized or moved to the other corner
    (a1,b1,c1,d1) = capture_map(cap,map_corner)
    new_centre = [int((a1+c1)/2),int((b1+d1)/2)]
    print(new_centre)
    dx = abs(box_centre[0]-new_centre[0])
    dy = abs(box_centre[1]-new_centre[1])
    if (map_corner=='right'):
        if (new_centre[0]>1180)and(new_centre[0]<1220)and(box_centre[1]>600):
        #image stabilization
        #if the new centre is significantly away from the old one(2 pixels horizontal and verdical)
            if ((dx > 3) and (dy > 3)):
                [a,b,c,d]=[a1,b1,c1,d1]
        else:
            map_corner = 'left'
            print('changed to left')
            (a,b,c,d) = capture_map(cap,map_corner)

    elif (map_corner=='left'):
        if (new_centre[0]>70)and(new_centre[0]<100)and (new_centre[1]>600):
            #image stabilization
            #if the new centre is significantly away from the old one(2 pixels horizontal and verdical)
                if ((dx > 3) and (dy > 3)):
                    [a,b,c,d]=[a1,b1,c1,d1]
        else:
            map_corner = 'right'
            print('changed to right')
            (a,b,c,d) = capture_map(cap,map_corner) 

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