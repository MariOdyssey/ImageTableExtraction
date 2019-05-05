import cv2
import numpy as np


img = cv2.imread("d:/WORK/0.jpg", 0)
orgimg = cv2.imread("d:/WORK/0.jpg")
h, w, z = orgimg.shape
vertical = np.zeros((h, w))
flip = np.zeros((h, w))

img = cv2.GaussianBlur(img, (3, 3), 0)
# edges = cv2.Canny(img, 50, 150, apertureSize=3)
ret3, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
img = ~img
lines = cv2.HoughLinesP(img, 0.8, np.pi / 180, 90, minLineLength=1, maxLineGap=5)  # 这里对最后一个参数使用了经验型的值
for line in lines:
    x1, y1, x2, y2 = line[0]
    # cv2.line(orgimg, (x1, y1), (x2, y2), (0, 255, 0), 2, lineType=cv2.LINE_AA)
    slope = (y2 - y1)/(x2 - x1)+0.0000000001
    if slope>-0.26 and slope<0.26:  # 水平
        cv2.line(flip, (x1, y1), (x2, y2), (255, 255, 255), 2, lineType=cv2.LINE_AA)
    if slope>3.73 or slope<-3.73:  # 水平
        cv2.line(vertical, (x1, y1), (x2, y2), (255, 255, 255), 2, lineType=cv2.LINE_AA)

cv2.imshow('flip', flip)
cv2.imshow('bertical', vertical)
cv2.imshow('org', orgimg)

points = cv2.bitwise_and(flip, vertical)
box = cv2.bitwise_or(flip, vertical)
cv2.imshow('and', points)
cv2.imshow('or', box)
cv2.waitKey(0)

