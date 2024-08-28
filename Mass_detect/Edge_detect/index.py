import cv2  
import numpy as np  
  
def canny_edge_detection(image_path, output_path, low_threshold, high_threshold):  
    # 读取图像  
    img = cv2.imread(image_path)  
    if img is None:  
        print("Error: Image cannot be read.")  
        return  
  
    # 转换为灰度图像  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
  
    # 应用高斯模糊以去除图像噪声  
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  
  
    # 使用Canny算法进行边缘检测  
    edges = cv2.Canny(blurred, low_threshold, high_threshold)  
  
    # 显示原图与边缘检测结果（可选）  
    # cv2.imshow('Original Image', img)  
    # cv2.imshow('Canny Edges', edges)  
    # cv2.waitKey(0)  # 等待按键，但这里只是为了显示，不是必须的  
  
    # 保存边缘检测结果到文件  
    cv2.imwrite(output_path, edges)  
    print(f"Edge detection result saved to {output_path}")  
  
    # 关闭所有窗口  
    # cv2.destroyAllWindows()  
  
# 调用函数  
canny_edge_detection('./images/mask1.jpg', './output/mask1.jpg', 100, 200)