import cv2

# 读取图像
image = cv2.imread('./images/mask3_hq.jpg')

# 将图像转换为灰度
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 对灰度图像进行二值化
_, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
# _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)

# 执行图像细化
thinned_image = cv2.ximgproc.thinning(binary_image)

# 保存图像
cv2.imwrite('./output/demo1_mask3.jpg',thinned_image)
