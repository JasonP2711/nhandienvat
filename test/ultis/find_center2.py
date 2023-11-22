import cv2
import numpy as np
from ultis.processing_image import contrast_stretching

def find_center2(gray_img,bboxes, low_clip,high_clip,intensity_of_template_gray):
    x1,y1,w,h = bboxes[0], bboxes[1], bboxes[2], bboxes[3]
    imgRoi = gray_img[y1-2:y1+h + 2,x1-2:x1+w + 2]
    padded_roi_gray = gray_img[y1-5:y1+h + 5,x1-5:x1+w + 5]
    thresholdImg = contrast_stretching(imgRoi,  low_clip,high_clip)
    padded_thresholdImg = contrast_stretching(imgRoi,  low_clip,high_clip)
    _,thresholdImg = cv2.threshold(thresholdImg,100,255, cv2.THRESH_BINARY_INV)
    _,padded_roi_gray = cv2.threshold(padded_thresholdImg,100,255, cv2.THRESH_BINARY_INV)
    intensity_of_roi_gray = np.sum(padded_roi_gray == 0)
    possible_grasp_ratio = (intensity_of_template_gray / intensity_of_roi_gray) * 100

    contours,_ = cv2.findContours(thresholdImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=len)

    s = cv2.contourArea(largest_contour)
    if s > 4000:
        
        # Tính moments của đường viền được chọn
        moments = cv2.moments(largest_contour)

    # Tính tọa độ tâm (x, y)
        center_x = int(moments['m10'] / moments['m00'])
        center_y = int(moments['m01'] / moments['m00'])
        cv2.circle(imgRoi, (int(center_x),int(center_y)), 1, (0,0,255))
        cv2.imwrite("out.jpg", imgRoi)
        center_obj = (center_x + x1, center_y + y1)
        return center_obj, possible_grasp_ratio
    
    else:
        print("Không có contour nào đáp ứng để lấy tâm!")
        return
        

