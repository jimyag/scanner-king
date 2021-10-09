import cv2

if __name__ == '__main__':
    image = cv2.imread('img.png')
    cv2.imshow("s",image)
    cv2.waitKey(0)
