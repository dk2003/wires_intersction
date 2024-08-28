# 图片细化（骨架提取）单张图片处理
import cv2
from skimage import morphology
import numpy as np

img = cv2.imread('./images/mask2.jpg', 0)     # 导入图片
_, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)   # 二值化处理
binary[binary==255] = 1
skel, distance =morphology.medial_axis(binary, return_distance=True)  # 图片细化（骨架提取）
dist_on_skel = distance * skel
dist_on_skel = dist_on_skel.astype(np.uint8)*255
cv2.imwrite("./output/demo3_mask2.jpg", dist_on_skel)           # 保存骨架提取后的图片
