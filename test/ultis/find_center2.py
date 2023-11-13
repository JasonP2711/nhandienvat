import cv2

def find_center2(img,bboxes):
    x1,y1,w,h = bboxes[0], bboxes[1], bboxes[2], bboxes[3]
    print("thong tin tam: ",x1,y1,w,h)
    imgRoi = img[y1:y1+h,x1:x1+w]
    imgRoiGray = cv2.cvtColor(imgRoi,cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(imgRoiGray, (3, 3), cv2.BORDER_DEFAULT)
    _,thresholdImg = cv2.threshold(blurred_image,100,255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("outsohoi.jpg",blurred_image)


    

