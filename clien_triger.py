import struct
import socket

# 定义格式字符串
fmt = ">4sIIBfB6s"
start_seq =b'star'
cmd_id = 1
args_len = 11
pallet_type = 1
depth_hint = 0.5
filter_mask = 1  # 00000001b
stop_seq = b'stop\r\n'

# 打包数据
data = struct.pack(fmt, start_seq, cmd_id, args_len, pallet_type, depth_hint, filter_mask, stop_seq)

# 连接服务端并发送数据
host = '127.0.0.1'  # 远程主机 IP
port = 2000  # 远程主机端口号
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((host, port))


# 接收服务端响应

while 1:
    sock.sendall(data)
    try:
        response = sock.recv(1024)  # 接收数据，每次最多接收 1024 字节
        # 解析接收到的数据
        start_seq = response[0:4]
        cmd_id = struct.unpack('!i', response[4:8])[0]
        err_code = struct.unpack('!i', response[8:12])[0]
        payload_len = struct.unpack('!i', response[12:16])[0]
        payload_data = struct.unpack('!ffffffffffffff', response[16:payload_len+10])
        stop_seq = response[payload_len + 10:]
        print('Response received:')
        print('Start sequence:', start_seq)
        print('Command ID:', cmd_id)
        print('Error code:', err_code)
        print('Payload length:', payload_len)
        print('Payload data:', payload_data)
        print('Stop sequence:', stop_seq)
    except:
        pass

# 关闭套接字
sock.close()
