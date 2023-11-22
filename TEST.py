import cv2
import numpy as np

def scale_object(image_path, output_path, scale_factor):
    # Đọc ảnh từ đường dẫn
    image = cv2.imread(image_path)

    # Kích thước ảnh
    height, width = image.shape[:2]

    # Tạo ma trận biến đổi affine
    scaling_matrix = np.array([[scale_factor, 0, 0], [0, scale_factor, 0]], dtype=np.float32)

    # Áp dụng phép biến đổi affine
    scaled_image = cv2.warpAffine(image, scaling_matrix, (width, height))

    # Lưu ảnh sau khi phóng to hoặc thu nhỏ
    cv2.imwrite(output_path, scaled_image)

# Đường dẫn đến ảnh đầu vào và đầu ra
input_image_path = r"E:\My_Code\NhanDienvat\test\template.jpg"
output_image_path = r"E:\My_Code\NhanDienvat\TEST.jpg"

# Hệ số phóng to hoặc thu nhỏ (ví dụ: 1.5 là phóng to 1.5 lần, 0.5 là thu nhỏ 0.5 lần)
scale_factor = 0.6

# Gọi hàm phóng to hoặc thu nhỏ đối tượng trong ảnh
scale_object(input_image_path, output_image_path, scale_factor)
