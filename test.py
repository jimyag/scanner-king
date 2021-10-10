import cv2
import imutils

from pyimagesearch.transform import four_point_transform

cap = cv2.VideoCapture(1)

while True:
    ret, image = cap.read()
    if ret:
        ratio = image.shape[0] / 500.0
        orig = image.copy()
        # image = imutils.resize(image, height=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edged = cv2.Canny(gray, 75, 200)
        screen_cnt = None
        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        for c in cnts:
            # 获得大致轮廓
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # 如果我们的近似轮廓有四点，那么我们
            # 可以假设我们已经找到了我们的目标
            if len(approx) == 4:
                screen_cnt = approx
                break
        if screen_cnt is not None:
            warped = four_point_transform(orig, screen_cnt.reshape(4, 2) * ratio)
            cv2.imshow("1", warped)

        cv2.imshow("src", image)
        if cv2.waitKey(1) == ord('q'):
            break
