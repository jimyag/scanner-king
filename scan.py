#! /usr/bin/env python3

from pyimagesearch.transform import four_point_transform
import argparse
import cv2
import imutils
import numpy as np

# requirements.txt

# 命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
args = vars(ap.parse_args())


# 计算原图高度比例
# 重新 resize
image = cv2.imread(args["image"])
ratio = image.shape[0] / 1000.0
orig = image.copy()

# 把图片 resize 到和屏幕一样大，并且保持比例
image = imutils.resize(image, height=1000)

# 全局变量存储点击的点
points = []

POINT_SIZE = 5
POINT_COLOR = (0, 255, 0)
LINE_COLOR = (0, 0, 255)
LINE_THICKNESS = 2

old_image = image.copy()


# 鼠标回调函数
def get_points(event, x, y, flags, param):
    global points, image
    if event == cv2.EVENT_LBUTTONDOWN:  # 检测到左键点击
        if len(points) < 4:  # 只允许点击 4 个点
            points.append((x, y))
            # 画点
            cv2.circle(image, (x, y), POINT_SIZE, POINT_COLOR, -1)

            # 如果有多个点，画线连接它们
            if len(points) > 1:
                cv2.line(image, points[-2], points[-1], LINE_COLOR, LINE_THICKNESS)
            # 如果是第四个点，将其与第一个点相连
            if len(points) == 4:
                cv2.line(image, points[3], points[0], LINE_COLOR, LINE_THICKNESS)
            cv2.imshow("Image", image)


cv2.imshow("Image", image)

# 设置鼠标回调函数
cv2.setMouseCallback("Image", get_points)

# 等待四个点被点击
while True:
    if len(points) >= 4:  # 如果已点击四个点
        break
    cv2.waitKey(1)

screen_cnt = np.array(points, dtype=np.int32)

cv2.drawContours(image, [screen_cnt], -1, (0, 255, 0), 2)

warped = four_point_transform(orig, screen_cnt.reshape(4, 2) * ratio)
cv2.imshow("Scanned", imutils.resize(warped, height=1000))
cv2.imwrite("scanned.jpg", warped)
cv2.waitKey(0)
