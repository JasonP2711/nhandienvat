import cv2
import numpy as np
import inspect

# def with_params(func):
#     def wrapper(img, params):
#         return func(img, **{k: params[k] for k in inspect.signature(func).parameters.keys() if k in params})
#     return wrapper

# @with_params
def contrast_stretching(img, low_clip, high_clip):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    low_val, high_val = np.percentile(img, (low_clip, high_clip))
    out_img = np.uint8(np.clip((img - low_val) * 255.0 / (high_val - low_val), 0, 255))
    return out_img

#Min - Max stretching
# def contrast_stretching(image):
#     image_cs = np.zeros((image.shape[0],image.shape[1]),dtype = 'uint8')
#     min = np.min(image)
#     max = np.max(image)
#     for i in range(image.shape[0]):
#       for j in range(image.shape[1]):
#         image_cs[i,j] = 255*(image[i,j] - min)/(max - min)

#     return image_cs
     



# /////////////Khái niêm 

# Phân vị (percentiles) thường được sử dụng trong kỹ thuật xử lý ảnh gọi là "contrast stretching" (mở rộng độ tương phản). Contrast stretching giúp cải thiện độ tương phản của ảnh bằng cách làm cho các giá trị pixel trong ảnh mở rộng từ một khoảng giá trị nhất định.

# Trong contrast stretching, phân vị được sử dụng để xác định giới hạn của khoảng giá trị pixel mới sau khi áp dụng phép biến đổi. Phép biến đổi này giúp đưa các giá trị pixel gần phân vị thấp về 0 và giá trị pixel gần phân vị cao về 255 (đối với ảnh 8-bit).