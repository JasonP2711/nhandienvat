from flask import Flask, request
import numpy as np
from components import *
from ultis import *
from copy import deepcopy
from API_flask import create_app
import csv
import time
import threading
# import keyboard

logger = logging.getLogger(__name__)

app = create_app()

@app.route('/cvu_process', methods=['GET','POST'])
def cvu_process():

  start_time = time.time()
# /////////Input////////////////////
  if request.method == "POST":
      #///Form data
      imgLink = request.form.get('imgLink')
      templateLink = request.form.get('templateLink')
      modelLink = request.form.get('modelLink')
      homographyLink = request.form.get('homographyLink')
      pathSaveOutputImg = request.form.get('pathSaveOutputImg')

      try:
            csvLink = request.form.get('csvLink')
            outputImgLink = request.form.get('outputImgLink')
            min_modify = int(request.form.get('min_modify'))
            max_modify = int(request.form.get('max_modify'))
            configScore = float(request.form.get('configScore'))
            img_size = int(request.form.get('img_size'))
            method = request.form.get('method')
            server_ip = request.form.get('server_ip')

      except Exception as e:
            logger.error(f'{e}\n')
            return f'{e}\n'

      #/////////Begin process/////////////////
      imgLink = imgLink.replace('\\', '/')
      templateLink = templateLink.replace('\\', '/')
      img = cv2.imread(imgLink)
      template = cv2.imread(templateLink)
      template = cv2.resize(template, (255,165))
      gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
      copy_of_template_gray = deepcopy(template_gray)
      minus_modify_angle = np.arange(0, min_modify, -2) 
      plus_modify_angle = np.arange(0, max_modify, 2) 
      low_clip, high_clip=5.0, 97.0
      copy_of_template_gray = contrast_stretching(copy_of_template_gray,  low_clip,high_clip)
      _, copy_of_template_gray = cv2.threshold(copy_of_template_gray, 100, 255, cv2.THRESH_BINARY_INV)
      # cv2.imwrite("thresTemp.jpg",copy_of_template_gray)
      intensity_of_template_gray = np.sum(copy_of_template_gray == 0)
      findCenter_type = 0
      good_points = []
      try:
            object_item = proposal_box_yolo(imgLink,modelLink,img_size,configScore)#object_item sẽ gồm list thông tin góc và tọa độ của đường bao
           
            if object_item == None:
                 return good_points
            #so luong phan tu da tim duoc
            num_obj1 = [2]
            num_obj2 = [2]
            # good_points.append([object_item[0][2]])
            for angle,bboxes,_ in object_item:
                #   print("------------------------------------------------------------")
                  result_queue = []
                  minus_sub_angles, plus_sub_angles = angle + minus_modify_angle, angle + plus_modify_angle
                  
                  # threshold = 0.95
                  point = match_pattern(gray_img, template_gray, bboxes, angle, eval(method)) 
                  if point is None:
                        continue
                  p1 = threading.Thread(target=find_center2, args=(gray_img,bboxes,low_clip,high_clip, intensity_of_template_gray, findCenter_type,result_queue,))
                  p2 = threading.Thread(target=compare_angle, args=(point,minus_sub_angles,plus_sub_angles, gray_img, template_gray, bboxes, angle, eval(method),result_queue,))
                  p1.start(), p2.start()                       
                  p1.join(), p2.join()
                  
                  # if len(result_queue) == 2:
                  bestAngle, bestPoint = result_queue[1]
                  center_obj, possible_grasp_ratio =  result_queue[0]
                  if center_obj[0] == None and center_obj[1] == None:
                        continue          
                  if possible_grasp_ratio < 90:
                        print("score<90!")
                        continue
                  good_points.append([center_obj,bestAngle,possible_grasp_ratio])
                  # Viết chữ lên hình ảnh
                  # cv2.putText(img, f"{bestAngle}", (int(center_obj[0]),int(center_obj[1] )), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                  # cv2.putText(img, f"({center_obj[0]},{center_obj[1]})", (int(center_obj[0]),int(center_obj[1] )), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
                  # x2, y2 = int(center_obj[0] + 150* np.cos(np.radians(bestAngle)) ),int(center_obj[1]  + 150* np.sin(np.radians(bestAngle)) )
                  # x3, y3 = int(center_obj[0] + 100* np.cos(np.radians(bestAngle+90)) ),int(center_obj[1]  + 100* np.sin(np.radians(bestAngle+90)) )
                  # cv2.line(img,(center_obj[0],center_obj[1] ),(x2,y2),(255,255,0),2)
                  # cv2.line(img,(center_obj[0],center_obj[1] ),(x3,y3),(255,0,255),2)
                  # cv2.imwrite("amTam.jpg",img)
                  # print("total: ",center_0,center_1,bestAngle,possible_grasp_ratio)
            print("good point arr: ",good_points)
            # create_homography(good_points)
            
            result = convert_point(good_points,homographyLink)
            result = result.tolist()
            result.insert(0,num_obj1)
            result.insert(0,num_obj2)
            print("result: ", result)
            print("time process: ",time.time() - start_time)
            return result
            # return []

      except Exception as e:
           print("System error: ", e)
           return []

#   if request.method == "GET":
#        return f'<div><h1>Get result</h1></div>'
       

if __name__ == "__main__":
     app.run(debug=True)