import cv2
import numpy as np

# Đọc ảnh từ file
image = cv2.imread(r'.\NDVat\train\images\20230919_104716_jpg.rf.747a0be5e965e191970e6a6c41c3ae68.jpg')

# Tọa độ của điểm (x, y) bạn muốn vẽ
x, y = 100, 100  # Thay thế x và y bằng tọa độ mong muốn

# Màu của điểm (BGR)
color = (0, 0, 255)  # Đỏ (Red)

# Kích thước của điểm
radius = 5  # Điểm có đường kính là 5 pixel

# Vẽ điểm lên ảnh
cv2.circle(image, (x, y), radius, color, -1)  # -1 để vẽ điểm đầy đủ (filled)

# Hiển thị ảnh với điểm đã vẽ
cv2.imshow('Image with Point', image)
cv2.waitKey(0)
cv2.destroyAllWindows()