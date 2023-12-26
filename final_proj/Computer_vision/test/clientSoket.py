import socket

# Khởi tạo socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Kết nối đến server
host = '192.168.1.9'
port = 4000
client_socket.connect((host, port))
print(f"Đã kết nối tới {host}:{port}")

# Gửi dữ liệu 1 byte đến server
# data = 1
number_to_send = 1
data = number_to_send.to_bytes(1, byteorder='big')
# client_socket.send(data)
client_socket.send(data)
print(f"Đã gửi dữ liệu: {data.decode('utf-8')}")

# Đóng kết nối
client_socket.close()
