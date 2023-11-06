import cv2
import numpy as numpy
from ultis.processing_image import *
import logging
logger = logging.getLogger(__name__)


def compute_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

# def find_center(img,bboxes,gray_img,intensity_of_template_gray):
#     # imgLink = r'E:\My_Code\test\dataset_specialItems\train\images\20230922_155058_jpg.rf.f9c8117af8571624c38fa89e5ba3b55c.jpg'
#     # img = cv2.imread(imgLink)
#     print("---------------------")
#     # tạo điểm cuối và tính tọa độ tâm 
#     (x1,y1,w,h ) = bboxes
    
#     x2,y2 = x1 + w,y1 + h
#     x_ctr,y_ctr = (x2 - x1)/2,(y2 - y1)/2
#     cv2.circle(img,(int(x_ctr),int(y_ctr)),2,(255, 255, 255) ,-1)
#     # print(x2,y2)
#     # print(x_ctr,y_ctr)
#     roi_gray = gray_img[y1:y2, x1:x2]
#     padded_roi_gray = gray_img[y1-100:y2+100, x1+100:x2+100]#còn lỗi ở đây, nếuvật nằm sát đường biên ảnh thì sẽ bị lỗi
#     # cv2.imwrite("abcgoc1.jpg",padded_roi_gray)
#     #Kéo giãn tương phản (Contrast Stretching) cho ảnh đã cắt:
    
#     # Định nghĩa tham số cho hàm contrast_stretching
#     params = {"low_clip": 10, "high_clip": 90}
#     roi_gray, padded_roi_gray = list(map(lambda x: contrast_stretching(x,  params), [roi_gray, padded_roi_gray]))
#     # cv2.imwrite("abc.jpg",roi_gray)
#     # cv2.imwrite("abcs.jpg",padded_roi_gray)
#     #chuyển ảnh sang ảnh nhị phân
#     #nếu muốn cải thiện, xem link: https://www.phamduytung.com/blog/2020-12-24-thresholding/
#     _,roi_gray = cv2.threshold(roi_gray,100,255,cv2.THRESH_BINARY_INV)
#     # cv2.imwrite("abc.jpg",roi_gray)
#     _, padded_roi_gray = cv2.threshold(padded_roi_gray,100,255,cv2.THRESH_BINARY_INV)
#     intensity_of_roi_gray = np.sum(padded_roi_gray == 0)
#     possible_grasp_ratio = (intensity_of_template_gray / intensity_of_roi_gray) * 100
#     #tìm các điểm cạnh để tìm được contour phù hợp:
#     #cv2.Canny() dùng để phát hiện biên trong hình ảnh
#     edges = cv2.Canny(roi_gray, 100, 200)
#     # cv2.dilate(src, kernel, iterations)  hàm trong thư viện OpenCV được sử dụng để thực hiện phép
#     #  mở rộng (dilation) trên hình ảnh nhị phân hoặc hình ảnh grayscale
#     edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
#      # tìm đường biên contour
#     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#     print("contour2: ",len(contours))

#     distances_circle = []
#     for contour in contours:
#         print("length of contour: ",len(contour))
        
        
#         #Kết quả của hàm cv2.minAreaRect(contour) là một tuple chứa thông tin về hình chữ nhật bao quanh:
#         # rect: Đây là hình chữ nhật bao quanh được biểu diễn dưới dạng tuple (center, size, angle):
#         # center: Tọa độ của trung tâm hình chữ nhật.
#         # size: Kích thước của hình chữ nhật, bao gồm chiều dài và chiều rộng.
#         # angle: Góc xoay của hình chữ nhật (được tính theo độ so với trục ngang).
#         # if cv2.minAreaRect(contour)[1][0] * cv2.minAreaRect(contour)[1][1] > roi_gray.shape[0] * roi_gray.shape[1] / 8:
#         #     continue
#         # try:
#         #     if (cv2.minAreaRect(contour)[1][0] / cv2.minAreaRect(contour)[1][1] > 1.1) or (cv2.minAreaRect(contour)[1][0] / cv2.minAreaRect(contour)[1][1] < 0.9):
#         #         continue
#         # except Exception as e:
#         #     # logger.exception(f'Filter contour: {e}\n')
#         #     continue

#         if len(contour) < 100: #fine tune
#             continue
      
        
#         # create circle from contour
#         (center_x_c, center_y_c), radious = cv2.minEnclosingCircle(contour)
        
#         distance_c = compute_distance((x_ctr, y_ctr), (center_x_c, center_y_c))
#         if distance_c > 25:
#             print("distance too long than x_ctr,y_ctr!!")
#             continue
#         print("there!")
#         print("quality center ",(center_x_c, center_y_c))
#         cv2.drawContours(img, contours, -1, (255, 0, 255), 2)
#         distances_circle.append({"center":(center_x_c, center_y_c), "radious":radious, "contour": contour})
        

#     # print("jndsflk: ",distances_circle,possible_grasp_ratio)
#     # return distances_circle,possible_grasp_ratio
#     distances_circle = sorted(distances_circle, key=lambda x:x["radious"])
#     # print("lo: ",distances_circle[0])
#     try:
#         centroid = np.mean(distances_circle[0]["contour"], axis=0)
#         centroid_x = centroid[0][0]
#         centroid_y = centroid[0][1]
        
#         center_c_x, center_c_y = distances_circle[0]["center"]  # center of circle    
#         true_center_x, true_center_y = (0.5*centroid_x + 0.5*center_c_x), (0.5*centroid_y + 0.5*center_c_y)
        
#     except Exception as e:
#         logger.error(f'No contour found\n')
#         true_center_x, true_center_y = w/2, h/2

#     center_obj = (true_center_x+x1, true_center_y+y1)
#     print("result score: ",possible_grasp_ratio)
#     return center_obj,possible_grasp_ratio,img
# ///////////////////////////////////////////////////////////
# Kéo giãn tương phản (Contrast Stretching) là một phương pháp quan trọng trong xử lý ảnh có nhiều
    # ứng dụng hữu ích. Dưới đây là một số lý do tại sao kéo giãn tương phản được sử dụng trong quy trình xử lý ảnh:

    # Tăng cường độ tương phản: Khi một hình ảnh có độ tương phản thấp, các chi tiết quan trọng có thể
    #  trở nên khó nhận biết. Kéo giãn tương phản giúp làm cho các đối tượng và chi tiết trong hình
    #  ảnh nổi bật hơn bằng cách làm cho các giá trị pixel gần nhau trải đều ra trong toàn bộ phạm vi
    #  giá trị pixel.

    # Cải thiện hình ảnh chụp trong điều kiện ánh sáng kém: Trong các tình huống ánh sáng yếu hoặc 
    # không đều, hình ảnh có thể trở nên mờ mịt hoặc bị mất độ tương phản. Kéo giãn tương phản có thể
    #  giúp làm rõ các chi tiết và cải thiện chất lượng hình ảnh.

    # Làm nổi bật các đặc trưng quan trọng: Trong nhiều tác vụ xử lý ảnh và phân tích đối tượng, việc
    #  kéo giãn tương phản có thể giúp làm nổi bật các đặc trưng quan trọng hoặc vùng quan tâm trong 
    # hình ảnh.

    # Phân đoạn ảnh: Trong việc phân đoạn (segmentation) hình ảnh để nhận diện đối tượng, kéo giãn 
    # tương phản có thể giúp làm cho các vùng quan trọng rõ ràng hơn, từ đó dễ dàng xác định ranh 
    # giới giữa các vùng.

    # Cải thiện hiển thị hình ảnh trên màn hình: Kéo giãn tương phản cũng thường được sử dụng để cải
    #  thiện hiển thị hình ảnh trên màn hình máy tính hoặc thiết bị khác để nó trông tốt hơn và dễ
    #  đọc hơn.
 #cv2.dilate(src, kernel, iterations) với src: Hình ảnh đầu vào, thường là một hình ảnh nhị
    #  phân (binary image) hoặc grayscale,kernel: Bộ lọc (kernel) được sử dụng để thực hiện phép
    #  mở rộng. Kernel là một ma trận có kích thước và hình dạng tùy ý, quyết định cách mở rộng 
    # được thực hiện, iterations: Số lần lặp (hoặc số lần áp dụng phép mở rộng). Nếu bạn đặt 
    # iterations là 1, thì phép mở rộng sẽ được áp dụng một lần duy nhất. là một



def find_center(bbox,img_gray , intensity_of_template_gray):
    (x1, y1, w, h) = bbox
    x2, y2 = x1 + w, y1 + h
    center_b_x, center_b_y = (x2-x1)/2, (y2-y1)/2
    
    roi_gray = img_gray[y1 - 50:y2 + 50, x1 - 50:x2 + 50]

    padded_roi_gray = img_gray[y1:y2, x1:x2]
    
    roi_gray, padded_roi_gray = list(map(lambda x: contrast_stretching(x, {"low_clip": 10, "high_clip": 90}), [roi_gray, padded_roi_gray]))
    _, roi_gray = cv2.threshold(roi_gray, 100, 255, cv2.THRESH_BINARY_INV)
    _, padded_roi_gray = cv2.threshold(padded_roi_gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    intensity_of_roi_gray = np.sum(padded_roi_gray == 0)
    cv2.imwrite('intensity_rois.jpg', padded_roi_gray)
    possible_grasp_ratio = (intensity_of_template_gray / intensity_of_roi_gray) * 100
    # print("score: ",possible_grasp_ratio)
    
    # find canny edges
    edges = cv2.Canny(roi_gray, 100, 200)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    
    # find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    distances_circle = []
    for contour in contours:
        if cv2.minAreaRect(contour)[1][0] * cv2.minAreaRect(contour)[1][1] > roi_gray.shape[0] * roi_gray.shape[1] / 8:
            continue

        try:
            if (cv2.minAreaRect(contour)[1][0] / cv2.minAreaRect(contour)[1][1] > 1.1) or (cv2.minAreaRect(contour)[1][0] / cv2.minAreaRect(contour)[1][1] < 0.9):
                continue
        except Exception as e:
            # logger.exception(f'Filter contour: {e}\n')
            continue

        if len(contour) < 100: #fine tune
            continue
        
        # create circle from contour
        (center_x_c, center_y_c), radious = cv2.minEnclosingCircle(contour)
        
        distance_c = compute_distance((center_b_x, center_b_y), (center_x_c, center_y_c))
        if distance_c > 25:
            continue
        print("qua!!")
        distances_circle.append({"center":(center_x_c, center_y_c), "radious":radious, "contour": contour})
    
    distances_circle = sorted(distances_circle, key=lambda x:x["radious"])
    
    try:
        centroid = np.mean(distances_circle[0]["contour"], axis=0)
        centroid_x = centroid[0][0]
        centroid_y = centroid[0][1]
        
        center_c_x, center_c_y = distances_circle[0]["center"]  # center of circle    
        true_center_x, true_center_y = (0.5*centroid_x + 0.5*center_c_x), (0.5*centroid_y + 0.5*center_c_y)
        
    except Exception as e:
        logger.error(f'No contour found\n')
        true_center_x, true_center_y = w/2, h/2

    center_obj = (true_center_x+x1, true_center_y+y1)

    return center_obj, possible_grasp_ratio
