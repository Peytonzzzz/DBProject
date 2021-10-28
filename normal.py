import cv2
import random
import numpy as np
xdata=[]
ydata=[]

with open('/Users/peytonzhu/Desktop/research/ddb1_v02_01.txt') as f:
    line=f.readlines()
    for values in line:
        xdata.append(values.split(',')[0])
        ydata.append(values.split(',')[1])
    for i in range(0,len(ydata)):
        ydata[i]=ydata[i].rstrip('\n')
    xdata = list(map(int, xdata))
    ydata =list(map(int,ydata))
listX=[]
listY=[]
print(xdata)
print(ydata)
def randomListX(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        numbers = random.randint(start, stop)
        if (numbers - 13 < 0 or numbers + 13 < 0):
            numbers = random.randint(start, stop)
        for x in range(len(xdata)):
            if (numbers <= xdata[x] + 20 or numbers >= xdata[x]) - 20:
                numbers = random.randint(start, stop)
                if (numbers - 13 < 0 or numbers + 13 < 0):
                    numbers = random.randint(start, stop)
        random_list.append(numbers)
    return random_list
def randomListY(start, stop, length):
    start, stop = (int(start), int(stop))
    length = int(abs(length))
    random_listY = []
    for i in range(length):
        numbers = random.randint(start, stop)
        if(numbers-13<1 or numbers+13<1):
            numbers = random.randint(start, stop)
        for x in range(len(xdata)):
             if (numbers <= ydata[x] + 20 or numbers >= ydata[x]- 20):
                numbers = random.randint(start, stop)
                if (numbers - 13 < 0 or numbers + 13 < 0):
                    numbers = random.randint(start, stop)
        random_listY.append(numbers)
    return random_listY

listX=randomListX(0,1500,120)
listY=randomListY(0,1152,120)
print(listX)
print(listY)

def pass_mask(mask, img):
    # qwe = reverse_image(img)
    qwe = img.copy()
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == 0:
                qwe[i][j] = 0
    # asd = cv2.filter2D(qwe, cv2.CV_8U, mask)
    return qwe
imgs=cv2.imread('/Users/peytonzhu/Desktop/GreenChannel.png')
#ret0, th0 = cv2.threshold(i, 0, 255, cv2.THRESH_BINARY)
#mask = cv2.erode(th0, np.ones((7, 7), np.uint8))
#j=pass_mask(mask,imgs)

for i in range(0,9):
    cropped = imgs[listY[i] - 13:listY[i] + 13, listX[i] - 13:listX[i] + 13]
    save_dir="/Users/peytonzhu/Desktop/Normal/normal6_{}.png".format(i)
    cv2.imwrite(save_dir,cropped)

