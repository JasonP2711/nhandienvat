import socket

# Khởi tạo socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Liên kết socket với địa chỉ và cổng
host = '127.0.0.1'
port = 48951
server_socket.bind((host, port))

# Lắng nghe kết nối từ client
server_socket.listen(1)
print(f"Đang lắng nghe kết nối tại {host}:{port}...")

# Chấp nhận kết nối từ client
client_socket, client_address = server_socket.accept()
print(f"Đã kết nối từ {client_address}")

# Nhận và in ra dữ liệu từ client
data = client_socket.recv(1)
received_number = int.from_bytes(data, byteorder='big')
print(f"Số nguyên nhận được: {received_number}")
# print(f"Dữ liệu nhận được: {data.decode('utf-8')}")

# Đóng kết nối
client_socket.close()
server_socket.close()
