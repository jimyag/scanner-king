# import the necessary packages
import numpy as np
import cv2


def order_points_new(pts):
    # 根据他们的 x 坐标对点进行排序
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    # 从排序中获取最左和最右的点
    # x -点
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    if left_most[0, 1] != left_most[1, 1]:
        left_most = left_most[np.argsort(left_most[:, 1]), :]
    else:
        left_most = left_most[np.argsort(left_most[:, 0])[::-1], :]
    (tl, bl) = left_most
    if right_most[0, 1] != right_most[1, 1]:
        right_most = right_most[np.argsort(right_most[:, 1]), :]
    else:
        right_most = right_most[np.argsort(right_most[:, 0])[::-1], :]
    (tr, br) = right_most

    return np.array([tl, tr, br, bl], dtype="float32")


def four_point_transform(image, pts):
    # 获得积分的一致顺序并拆开它们
    # individually
    rect = order_points_new(pts)
    (tl, tr, br, bl) = rect

    # 计算新图像的宽度，这将是 右下角和左下角之间的最大距离 x坐标右上角和左上角 X 坐标
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # 计算新图像的高度，这将是右上角和右下角 y 坐标或左上角和左下角 y 坐标之间的最大距离
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # 现在，我们有新图像的尺寸，构造 一组目的地点，以获得"鸟瞰图"，（即自上而下视图）的图像，再次指定点在左上角、右上角、右下角和左下角次序
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # 将透视转换矩阵，然后应用它
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    # 扭曲
    return warped
