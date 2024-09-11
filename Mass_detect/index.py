import cv2
import os
import numpy as np

# 细化，前景为白，背景为黑
def thin(url):
    # 处理路径
    full_image_path = os.path.join(os.getcwd(), url)

    # 读取图像
    image = cv2.imread(full_image_path)

    # 转化为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化
    # _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)

    # 细化，要求前景是白色，背景是黑色
    thinned_image = cv2.ximgproc.thinning(binary_image)

    return thinned_image

# 获取 8 邻域像素
def neighbours_fn(x,y,image):
    """Return 8-neighbours of image point P1(x,y), in a clockwise order"""
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1], img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]   

# 获取交点
def getSkeletonIntersection(skeleton):
    validIntersection = [[0,1,0,1,0,0,1,0],[0,0,1,0,1,0,0,1],[1,0,0,1,0,1,0,0],
                         [0,1,0,0,1,0,1,0],[0,0,1,0,0,1,0,1],[1,0,0,1,0,0,1,0],
                         [0,1,0,0,1,0,0,1],[1,0,1,0,0,1,0,0],[0,1,0,0,0,1,0,1],
                         [0,1,0,1,0,0,0,1],[0,1,0,1,0,1,0,0],[0,0,0,1,0,1,0,1],
                         [1,0,1,0,0,0,1,0],[1,0,1,0,1,0,0,0],[0,0,1,0,1,0,1,0],
                         [1,0,0,0,1,0,1,0],[1,0,0,1,1,1,0,0],[0,0,1,0,0,1,1,1],
                         [1,1,0,0,1,0,0,1],[0,1,1,1,0,0,1,0],[1,0,1,1,0,0,1,0],
                         [1,0,1,0,0,1,1,0],[1,0,1,1,0,1,1,0],[0,1,1,0,1,0,1,1],
                         [1,1,0,1,1,0,1,0],[1,1,0,0,1,0,1,0],[0,1,1,0,1,0,1,0],
                         [0,0,1,0,1,0,1,1],[1,0,0,1,1,0,1,0],[1,0,1,0,1,1,0,1],
                         [1,0,1,0,1,1,0,0],[1,0,1,0,1,0,0,1],[0,1,0,0,1,0,1,1],
                         [0,1,1,0,1,0,0,1],[1,1,0,1,0,0,1,0],[0,1,0,1,1,0,1,0],
                         [0,0,1,0,1,1,0,1],[1,0,1,0,0,1,0,1],[1,0,0,1,0,1,1,0],
                         [1,0,1,1,0,1,0,0]]
    image = skeleton.copy()
    image = image/255
    intersections = list()
    for x in range(1,len(image)-1):
        for y in range(1,len(image[x])-1):
            # If we have a white pixel
            if image[x][y] == 1:
                neighbours = neighbours_fn(x,y,image)
                valid = True
                if neighbours in validIntersection:
                    intersections.append((y,x))
    # Filter intersections to make sure we don't count them twice or ones that are very close together
    for point1 in intersections:
        for point2 in intersections:
            if (((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) < 10**2) and (point1 != point2):
                intersections.remove(point2)
    # Remove duplicates
    intersections = list(set(intersections))
    return intersections

# 绘制交点并且保存
def intersections_detect(inputUrl,outputUrl):
    skeleton = thin(inputUrl)
    inverted_image = 255 - skeleton
    # 前景边黑、背景变白
    intersection_points = getSkeletonIntersection(skeleton)
    # 在图像上绘制交点
    for point in intersection_points:
        cv2.circle(inverted_image, (int(point[0]), int(point[1])), 10, (0, 255, 0), -1)
    cv2.imwrite(outputUrl, inverted_image)

if __name__=='__main__':
    intersections_detect('./input/mask.jpg','./output/intersection.jpg')

__all__ = ['intersections_detect']