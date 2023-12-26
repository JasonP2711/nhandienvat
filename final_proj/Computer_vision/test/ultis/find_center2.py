import cv2
import numpy as np
from ultis.processing_image import contrast_stretching



def find_center2(gray_img,bboxes, low_clip,high_clip,intensity_of_template_gray,findCenter_type,result_queue):

    # print("time excute: ", time.time())
    x1,y1,w,h = bboxes[0], bboxes[1], bboxes[2], bboxes[3]
    # c1,c2 = x1 + w/2, y1 + h/2
    center_obj = (0,0)
    if(x1<10 or y1 <10):
     gray_img = cv2.copyMakeBorder(gray_img,20,20, 20, 20,cv2.BORDER_CONSTANT, value=0 )
    imgRoi = gray_img[y1 - 3:y1+h +3,x1 - 3:x1+w +3]
    # cv2.imwrite("roi_findCenter.jpg",imgRoi)
    
    padded_roi_gray = gray_img[y1-20:y1+h + 20,x1-20:x1+w + 20]
    thresholdImg = contrast_stretching(imgRoi,low_clip,high_clip)
    padded_thresholdImg = contrast_stretching(padded_roi_gray,  low_clip,high_clip)
    _,thresholdImg = cv2.threshold(thresholdImg,100,255, cv2.THRESH_BINARY_INV)
    _,padded_roi_gray = cv2.threshold(padded_thresholdImg,100,255, cv2.THRESH_BINARY_INV)
    intensity_of_roi_gray = np.sum(padded_roi_gray == 0)
    possible_grasp_ratio = (intensity_of_template_gray / intensity_of_roi_gray) * 100
    if findCenter_type == 0:
        try:
            contours,_ = cv2.findContours(thresholdImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            check_true = False
            for element in contours:
                s = cv2.contourArea(element)
                # print('S contour: ',s)
                if s < 1500 or s > 3000:
                    continue
                check_true = True
                (x_axis, y_axis), radius = cv2.minEnclosingCircle(element)
                center = (int(x_axis + x1),int(y_axis + y1)) 
                radius = int(radius) 
                cv2.circle(gray_img,center,radius,(0,255,0),1) 
                cv2.circle(gray_img,center,1,(255,255,0),3) 
                center_obj = (center[0],center[1])
            if check_true == False:
                center_obj = (None, None)

        except Exception as e:
                # print("S < 1500 or s > 3000")
                center_obj = (None, None)
    result_queue.append((center_obj,possible_grasp_ratio))
    
    return center_obj, possible_grasp_ratio


