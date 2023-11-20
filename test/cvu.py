from flask import Flask, request
from components import *
from ultis import *
from ultralytics import YOLO
from copy import deepcopy
from API_flask import create_app
from time import time

logger = logging.getLogger(__name__)

app = create_app()

@app.route('/', methods = ['GET'])
def say_hello():
     logger.info('Xin chào, đây là log!')
     return '<h1>Hello World!!</h1>'

@app.route('/cvu_process', methods=['GET','POST'])
def cvu_process():
  model = YOLO('yolov8n-seg.pt')  # load an official model
  model = YOLO(r'E:\My_Code\NhanDienvat\runs\segment\train\weights\last.pt')  # load a custom model

# /////////Input////////////////////
  if request.method == "POST":
      #///Form data
      imgLink = request.form.get('imgLink')
      templateLink = request.form.get('templateLink')
      modelLink = request.form.get('modelLink')
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
      temp_height, temp_width = template.shape[0], template.shape[1]
      print("temp_height, temp_width: ",temp_height, temp_width)
      gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

      template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
      copy_of_template_gray = deepcopy(template_gray)
      minus_modify_angle = np.arange(-1, min_modify, -1) #-20
      plus_modify_angle = np.arange(1, max_modify, 1) #20
      low_clip=5
      high_clip=97
      copy_of_template_gray = contrast_stretching(copy_of_template_gray,  low_clip,high_clip)
      # copy_of_template_gray = contrast_stretching(copy_of_template_gray)
      cv2.imwrite("testTemp1.jpg",copy_of_template_gray)
      _, copy_of_template_gray = cv2.threshold(copy_of_template_gray, 100, 255, cv2.THRESH_BINARY_INV)
      cv2.imwrite('intensity_template.jpg', copy_of_template_gray)
      intensity_of_template_gray = np.sum(copy_of_template_gray == 0)

      object_item = proposal_box_yolo(imgLink,modelLink,img_size,configScore)#object_item sẽ gồm list thông tin góc và tọa độ của đường bao
      print("in4: ", object_item)

      good_points = []
      for angle,bboxes in object_item:
            center_obj,possible_grasp_ratio = find_center2(gray_img,bboxes,low_clip,high_clip, intensity_of_template_gray)
            cv2.circle(img, (center_obj[0],center_obj[1]), 1, (0,0,255))
            # center_obj,possible_grasp_ratio = find_center(bboxes, gray_img, intensity_of_template_gray)
            # cv2.circle(img,(int(center_obj[0]),int(center_obj[1])),2,(0, 0, 255) ,-1)
          
            if possible_grasp_ratio < 50:
                      print("score<50!")
                      continue
            print("score: ",possible_grasp_ratio )
            print("angle: ", angle)
            print("tam: ", center_obj)
            minus_sub_angles = angle + minus_modify_angle
            plus_sub_angles = angle + plus_modify_angle
            minus_length = len(minus_sub_angles)
            plus_length = len(plus_sub_angles)
            minus_pointer, minus_check = 0, False
            plus_pointer, plus_check = 0, False
            sub_minus_points = []
            sub_plus_points = []
          
            threshold = 0.95
            point = match_pattern(gray_img, template_gray, bboxes, angle, method, threshold)
            print("zzz: ",point)
            cv2.putText(img, (f"({int(center_obj[0])},{int(center_obj[1])})"),(int(center_obj[0])+10,int(center_obj[1]+10)) , cv2.FONT_HERSHEY_SIMPLEX,0.4,(255, 255, 255), 2)
            cv2.putText(img, (f"({round(angle,2)})"),(int(center_obj[0])+10,int(center_obj[1]+20)) , cv2.FONT_HERSHEY_SIMPLEX,0.4,(255, 150, 255), 2)

            minus_sub_angles = angle + minus_modify_angle
            plus_sub_angles = angle + plus_modify_angle
            minus_length = len(minus_sub_angles)
            plus_length = len(plus_sub_angles)
              
            minus_pointer, minus_check = 0, False
            plus_pointer, plus_check = 0, False
            sub_minus_points = []
            sub_plus_points = []
              
            if point is None:
                continue
            while True:
                      if (minus_length == 0 and plus_length == 0):
                              break

                      if minus_length == 0 or minus_pointer >= minus_length:
                              minus_check = True
                      elif plus_length == 0 or plus_pointer >= plus_length:
                              plus_check = True

                      if not minus_check and minus_length != 0:
                              print("---")
                              minus_point = match_pattern(gray_img, template_gray, bboxes, minus_sub_angles[minus_pointer], method, threshold)
                              if minus_point is not None:
                                minus_check = minus_point[4] < point[4] if minus_pointer == 0 else minus_point[4] < sub_minus_points[-1][4]
                              else:
                                minus_check = True
                              
                              if not minus_check:
                                sub_minus_points.append(minus_point)
                              minus_pointer += 1
                      
                      if not plus_check and plus_length != 0:
                              print("+++")
                              plus_point = match_pattern(gray_img, template_gray, bboxes, plus_sub_angles[plus_pointer], method, threshold)
                              if plus_point is not None:
                                plus_check = plus_point[4] < point[4] if plus_pointer == 0 else plus_point[4] < sub_plus_points[-1][4]
                              else:
                                plus_check = True
                              
                              if not plus_check:
                                sub_plus_points.append(plus_point)
                              plus_pointer += 1
                      
                      if minus_check and plus_check:
                        break
              
            best_minus_point = sub_minus_points[-1] if sub_minus_points else None
            best_plus_point = sub_plus_points[-1] if sub_plus_points else None
              
            if (best_minus_point is not None) and (best_plus_point is not None):
                best_point = best_minus_point if best_minus_point[4] > best_plus_point[4] else best_plus_point
            elif (best_minus_point is None) and (best_plus_point is None):
                best_point = point
            else:
                best_point = best_minus_point or best_plus_point
              
            if point:
                good_points.append((best_point[2], center_obj, possible_grasp_ratio))
            print("good point: ",good_points)

      
      cv2.imwrite("amTam.jpg",img)
 
      # resize(imgLink,pathSaveOutputImg)
      return f'<div><h1>Result: </h1><p>{object_item}</p></div>'
  if request.method == "GET":
       return f'<div><h1>Get result</h1></div>'
       

if __name__ == "__main__":
     app.run(debug=True)