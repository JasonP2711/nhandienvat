# server.py
import socket
import numpy as np
import struct
import requests
import time

# Định nghĩa host và port mà server sẽ chạy và lắng nghe
host = '172.30.160.1'
# host = '192.168.0.222'
port = 48951

headers = {"Content-Type": "application/json", "Authorization": "Bearer your_token"}

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host, port))
s.listen(1)  # Chỉ chấp nhận 1 kết nối đồng thời
print("Server listening on port", port)
# data_arr=[]




try:
    while True:
        c, addr = s.accept()
        print("Connected from", str(addr))

        # # Server sử dụng kết nối để gửi dữ liệu tới client dưới dạng binary
        # c.send(b"Hello, how are you")
        while True:
            # Nhận 1 byte dữ liệu từ client
            data = c.recv(1)
            # decode_data = struct.unpack('!f', data)[0]
            decode_data = int.from_bytes(data, byteorder='little')
            print(f"Received data: {decode_data}")
            if decode_data == 1:
                # for i in range(len(data_arr)):
                #     data_arr[i] = data_arr[i] + 10
                # for element in data_arr:
                    
                #     print("hjkhjk: ",element)
                #     byte_value = struct.pack('!1f', element)
                #     c.sendall(byte_value)
                #     print(f"Dữ liệu gui từ server: {byte_value}")
                #     # ////////////////////////////////
                api_url = "http://127.0.0.1:5000/cvu_process"
                form_data={
                    "imgLink" : r"E:\My_Code\NhanDienvat\final_proj\Computer_vision\test\xulyanh\xulyanh\31201927_12_08_17_26_39_651.jpg",
                    "templateLink" : r"E:\My_Code\NhanDienvat\final_proj\Computer_vision\test\template.jpg",
                    "modelLink" : r"E:\My_Code\NhanDienvat\final_proj\Computer_vision\test\runs_final\segment\train\weights\last.pt",
                    "pathSaveOutputImg" : "",
                    "csvLink" : "",
                    "outputImgLink" : "",
                    "min_modify" : "-10",
                    "max_modify" : "10",
                    "configScore" : "0.8",
                    "img_size" : "640",
                    "method" : "cv2.TM_CCORR_NORMED",
                    "server_ip" : ""
                }
                response = requests.post(api_url, data=form_data)
                if response.status_code == 200:
                    print("respons and type: ", response.json(), type(response))
                    data_arr = response.json()
                    # byte_value_length = struct.pack('!1f', len(data_arr))
                    # c.sendall(byte_value_length)
                    # print(f"Dữ liệu gui từ server: {byte_value_length}")
                    for object in data_arr:
                        for element in object:
                            print("hjkhjk: ",element,type(element))
                            if isinstance(element, float):
                                byte_value = struct.pack('!f', element)
                                c.sendall(byte_value)
                                print(f"Dữ liệu gui từ server: {byte_value}")
                            if isinstance(element, int):
                                byte_value = struct.pack('!i', element)
                                c.sendall(byte_value)
                                print(f"Dữ liệu gui từ server: {byte_value}")
                    print("done!!")
                    break
            # c.close()

except Exception as e:
    print("Error:", str(e))

finally:
    print("loi")
    s.close()
