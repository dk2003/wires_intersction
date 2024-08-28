from skimage.morphology import skeletonize
import cv2
import numpy as np

image = cv2.imread('./images/mask2.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用 cv2.threshold 进行二值化处理，阈值设为 128，将高于这个阈值的像素值设为 255，低于或等于这个阈值的像素值设为 0。
# 这里没有使用 cv2.THRESH_BINARY_INV，因此图像不会被反转。
_, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

# 将二值化图像的数据类型转换为布尔类型 (np.bool_)，这使得 skeletonize 函数能够正确处理。
binary_image = binary_image.astype(np.bool_)

# 取反，因为我们想要前景为True（1）
# 注意，前后景一定要搞清楚！！！！，否则将背景细化了...
# binary_image = ~binary_image

skeleton = skeletonize(binary_image)
# 将骨架化结果转换回uint8格式以保存
skeleton_uint8 = (skeleton * 255).astype(np.uint8)
cv2.imwrite('./output/demo4_mask2.jpg', skeleton_uint8)