import cv2
import numpy as np
from ultis.processing_image import contrast_stretching

def find_center2(gray_img,bboxes, low_clip,high_clip,intensity_of_template_gray):
    
    x1,y1,w,h = bboxes[0], bboxes[1], bboxes[2], bboxes[3]
    if(x1<10 or y1 <10):
     gray_img = cv2.copyMakeBorder(gray_img,10,10, 10, 10,cv2.BORDER_CONSTANT, value=0 )
    imgRoi = gray_img[y1 - 5:y1+h +5,x1 - 5:x1+w +5]
    
    cv2.imwrite("roi_findCenter.jpg",imgRoi)
    
    padded_roi_gray = gray_img[y1-5:y1+h + 5,x1-5:x1+w + 5]
    
    thresholdImg = contrast_stretching(imgRoi,low_clip,high_clip)
    cv2.imwrite("thes.jpg",thresholdImg)
    padded_thresholdImg = contrast_stretching(imgRoi,  low_clip,high_clip)
    _,thresholdImg = cv2.threshold(thresholdImg,100,255, cv2.THRESH_BINARY_INV)
    _,padded_roi_gray = cv2.threshold(padded_thresholdImg,100,255, cv2.THRESH_BINARY_INV)
    intensity_of_roi_gray = np.sum(padded_roi_gray == 0)
    possible_grasp_ratio = (intensity_of_template_gray / intensity_of_roi_gray) * 100
    try:
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
            
        
        else:
            center_obj = (x1 + w/2, y1 + h/2)
    except Exception as e:
            center_obj = (x1 + w/2, y1 + h/2)


    return center_obj, possible_grasp_ratio


