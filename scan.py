from skimage.filters import threshold_local

from pyimagesearch.transform import four_point_transform
import argparse
import cv2
import imutils

# 命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())



# 计算原图高度比例
# 重新resize
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

# 处理图片
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
edged = cv2.Canny(gray, 75, 200)

# 显示轮廓和原图
print("STEP 1: Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 找到最大的轮廓边
# 初始化屏幕轮廓

cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# loop over the contours
for c in cnts:
    # 获得大致轮廓
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # 如果我们的近似轮廓有四点，那么我们
    # 可以假设我们已经找到了我们的目标
    if len(approx) == 4:
        screen_cnt = approx
        break

# s显示纸张的轮廓（轮廓）
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screen_cnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 应用四点转换以获得自上而下
# view of the original image
warped = four_point_transform(orig, screen_cnt.reshape(4, 2) * ratio)

# 将扭曲的图像转换为灰度，然后对它进行阈值
# 黑白效果
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset=10, method="gaussian")
warped = (warped > T).astype("uint8") * 255


print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height=650))
cv2.imshow("Scanned", imutils.resize(warped, height=650))
cv2.waitKey(0)
