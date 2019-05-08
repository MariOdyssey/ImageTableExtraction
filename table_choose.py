import os
import cv2 as cv
import cv2
import numpy as np
import math


def findPoint(image):
    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    h, w = image.shape
    vertical = np.zeros((h, w), dtype=np.uint8)
    flip = np.zeros((h, w), dtype=np.uint8)

    img = cv.GaussianBlur(image, (3, 3), 0)
    ret3, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = ~img
    lines = cv2.HoughLinesP(img, 0.8, np.pi / 180, 90, minLineLength=50, maxLineGap=5)  # 这里对最后一个参数使用了经验型的值
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # cv2.line(orgimg, (x1, y1), (x2, y2), (0, 255, 0), 2, lineType=cv2.LINE_AA)
        slope = (y2 - y1) / (x2 - x1) + 0.0000000001
        if slope > -0.26 and slope < 0.26:  # 水平
            cv2.line(flip, (x1, y1), (x2, y2), (255, 255, 255), 4, lineType=cv2.LINE_AA)
        if slope > 3.73 or slope < -3.73:  # 水平
            cv2.line(vertical, (x1, y1), (x2, y2), (255, 255, 255), 4, lineType=cv2.LINE_AA)

    return flip, vertical


def tableEx(image):
    rows, cols, channels = image.shape
    image_copy = image.copy()

    # ##### 旋转校正Rotation #####
    # # 统计图中长横线的斜率来判断整体需要旋转矫正的角度
    # gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    # edges = cv.Canny(gray, 50, 150, apertureSize=3)  # 50,150,3
    # lines = cv.HoughLinesP(edges, 1, np.pi / 180, 300, 0, minLineLength=10, maxLineGap=5)  # 650,50,20
    # pi = 3.1415
    # theta_total = 0
    # theta_count = 0
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     rho = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    #     theta = math.atan(float(y2 - y1) / float(x2 - x1 + 0.001))
    #     if theta < pi / 4 and theta > -pi / 4:
    #         theta_total = theta_total + theta
    #         theta_count += 1
    #         cv.line(image_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #         # cv.line(edges, (x1, y1), (x2, y2), (0, 0, 0), 2)
    # theta_average = theta_total / theta_count
    #
    # affineShrinkTranslationRotation = cv.getRotationMatrix2D((0, rows), theta_average*180/pi, 1)
    # ShrinkTranslationRotation = cv.warpAffine(image, affineShrinkTranslationRotation, (cols, rows))
    # image = ShrinkTranslationRotation

    flip, vertical = findPoint(image)
    mask = flip + vertical

    joints = cv2.bitwise_and(flip, vertical)
    # 根据矩形大小筛选矩形框，并画在矫正后的表格上
    # cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    small_rects = []
    length = len(contours)
    for i in range(length):
        cnt = contours[i]
        area = cv.contourArea(cnt)
        approx = cv.approxPolyDP(cnt, 3, True)  # 3
        x, y, w, h = cv.boundingRect(approx)
        rect = (x, y, w, h)
        pia = int(max(rows, cols) / 50)
        left = x - pia
        right = x + w + pia
        top = y - pia
        bottow = y + h + pia
        if left < 0:
            left = 0
        if top < 0:
            top = 0
        if right > cols:
            right = cols
        if bottow > rows:
            bottow = rows
        roi = joints[top:bottow, left:right]
        joints_contours, joints_hierarchy = cv.findContours(roi, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        # h < 800 and h > 0 and w > 0 and len(joints_contours)<=15 and len(joints_contours)>=4:
        if len(joints_contours) <= 15 and len(joints_contours) >= 4:  # important
            # cv.rectangle(image, (x, y), (x+w, y+h), (255-h*3, h*3, 0), 3)
            small_rects.append(rect)

    return small_rects


if __name__ == '__main__':
    image = cv.imread("d:/WORK/30201.jpg")
    rect = tableEx(image)
    for i in rect:
        x, y, w, h = i
        cv.rectangle(image, (x, y), (x + w, y + h), (255 - h, h, 0), 3)

    cv2.namedWindow('', 0)
    cv2.resizeWindow('', 500, 500)
    cv2.imshow('', image)
    cv2.waitKey(0)
