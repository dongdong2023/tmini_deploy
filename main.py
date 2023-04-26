# encoding=utf-8
import time

from qtgui import Window
import sys
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QSplashScreen, QMessageBox, QFileDialog
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtCore import pyqtSignal
import qdarkstyle
from qdarkstyle.light.palette import LightPalette
from model import Model
from camera_utils import VITMINI
import numpy as np
import cv2
from get_angle import fit
import threading
import socket
import inspect
import struct
import ctypes
from conf_util import read_cof

def pack_data(*args):
    packed = b''
    for arg in args:
        if isinstance(arg, int):
            packed += struct.pack('!i', arg)
        elif isinstance(arg, float):
            packed += struct.pack('!f', arg)
        elif isinstance(arg, tuple):
            packed += struct.pack('!fff', *arg)
    return packed

def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""

    tid = ctypes.c_long(tid)

    if not inspect.isclass(exctype):
        exctype = type(exctype)

    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))

    if res == 0:

        raise ValueError("invalid thread id")

    elif res != 1:

        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)

        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


# 重写QSplashScreen类
class MySplashScreen(QSplashScreen):
    # 鼠标点击事件
    def mousePressEvent(self, event):
        pass


class MainWindow(Window):
    global_button_status = True
    client_socket_list = list()
    # 自定义消息
    dialogSignel1 = pyqtSignal(str)  # textbrowser
    dialogSignel2 = pyqtSignal(str)  # textbrowser_2

    def __init__(self):
        super(MainWindow, self).__init__()
        # 按钮
        self.pushButton_4.clicked.connect(self.start_det)
        self.pushButton.clicked.connect(self.file_path)
        self.pushButton_2.clicked.connect(self._init)
        self.dialogSignel1.connect(self.updata_textbrowser1)
        self.dialogSignel2.connect(self.updata_textbrowser2)
        self.read()
        self.pushButton_2.click()

    def read(self):  # 读取
        cfg = read_cof()
        serv_ip = cfg['Server']['ip']
        post = cfg['Server']['post']
        output = cfg['Server']['output']
        cam_ip = cfg['Camera']['ip']
        type = cfg['Camera']['type']
        disk = cfg['Model']['disk']

        self.lineEdit_11.setText(str(cam_ip))
        if type=='trigger':
            self.radioButton_6.setChecked(True)
        else:
            self.radioButton_5.setChecked(True)

        self.lineEdit_6.setText(str(serv_ip))
        self.lineEdit_7.setText(str(post))
        if output=='one':
            self.radioButton_3.setChecked(True)
        else:
            self.radioButton_4.setChecked(True)

        if disk=='cpu':
            self.radioButton.setChecked(True)
        else:
            self.radioButton_2.setChecked(True)


    def _init(self):
        result = self.get_data()
        if result is None:
            return
        # 相机初始化
        try:
            self.vit = VITMINI(result[0][1])
            self.vit.init_camera()
        except Exception as e:
            self.dialogSignel1.emit(str(e))
            QMessageBox.critical(self, '错误', '相机问题')
        # 模型初始化
        try:
            self.models = Model()
            self.models.init_model(result[2][0], result[2][5])
        except Exception as e:
            self.dialogSignel1.emit(str(e))
            QMessageBox.critical(self, '错误', '模型问题')
        # 通讯初始化
        try:
            self.tcp_server_start(result[3][1], result[3][2])
        except Exception as e:
            self.dialogSignel1.emit(str(e))
            QMessageBox.critical(self, '错误', '模型问题')

        self.dialogSignel1.emit('初始化完成')

    def file_path(self):
        fname = QFileDialog.getOpenFileName(self, '打开文件', )
        if fname[0]:
            try:
                self.lineEdit.setText(fname[0])
            except Exception as e:
                self.dialogSignel1.emit(str(e))
                QMessageBox.critical(self, '错误', '打开文件失败，可能是文件内型错误')

    def tcp_server_start(self, ip, post):
        """
        功能函数，TCP服务端开启的方法
        :return: None
        """
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 取消主动断开连接四次握手后的TIME_WAIT状态
        self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            address = (ip, int(post))
            self.tcp_socket.bind(address)
            self.tcp_socket.listen()
            # 设定套接字为非阻塞式
            self.tcp_socket.setblocking(False)
            self.sever_th = threading.Thread(target=self.tcp_server_concurrency)
            self.sever_th.start()
            self.dialogSignel1.emit('TCP服务端正在监听端口:%s\n' % str(self.lineEdit_robot_post.text()))
        except Exception as ret:
            msg = '请检查IP，端口\n'
            self.dialogSignel1.emit(msg)

    def tcp_server_concurrency(self):
        """
        功能函数，供创建线程的方法；
        使用子线程用于监听并创建连接，使主线程可以继续运行，以免无响应
        使用非阻塞式并发用于接收客户端消息，减少系统资源浪费，使软件轻量化
        :return:None
        """
        while True:
            try:
                client_socket, client_address = self.tcp_socket.accept()
            except Exception as e:
                pass  # 因为是非堵塞，所以有可能会出现socket异常的情况
            else:
                client_socket.setblocking(False)
                # 将创建的客户端套接字存入列表,client_address为ip和端口的元组
                self.client_socket_list.append((client_socket, client_address))
                msg = 'TCP服务端已连接IP:%s端口:%s\n' % client_address
                self.dialogSignel1.emit(msg)
                # 轮询客户端套接字列表，接收数据
            for client, address in self.client_socket_list:
                try:
                    recv_msg = client.recv(1024)
                except Exception as e:
                    pass
                else:
                    if recv_msg:
                        if len(recv_msg)==22:
                            # 定义格式字符串
                            fmt = ">4sIiI6s"
                            start_seq = bytes((115, 116, 97, 114))  # b'star'
                            cmd_id = 0
                            error_code = 0  # PDS_NO_ERRORS
                            payload_len = 6
                            stop_seq = bytes((115, 116, 111, 112, 13, 10))  # b'stop\r\n'
                            # 打包数据
                            data = struct.pack(fmt, start_seq, cmd_id, error_code, payload_len, stop_seq)
                            self.tcp_send(data)
                        else:
                            self.start_det()

                    else:
                        client.close()
                        self.client_socket_list.remove((client, address))

    def tcp_send(self, msg):
        try:
            for client, address in self.client_socket_list:
                client.sendall(msg)
            msg = 'TCP服务端已发送{}\n'.format(msg)
            self.dialogSignel1.emit(msg)
        except Exception as e:
            self.dialogSignel1.emit(str(e))
            self.client_socket_list.remove((client, address))

    def get_data(self):
        try:
            trige_select = self.trige_select.checkedButton().text()
        except:
            QMessageBox.warning(self, '警告', '相机设置必须选一个', QMessageBox.Yes | QMessageBox.No)
            return
        camera_ip = self.lineEdit_11.text()
        data_distance = self.checkBox.isChecked()
        saveimg = self.checkBox_3.isChecked()
        savepcl = self.checkBox_4.isChecked()
        # roi
        roi_x = self.lineEdit_12.text()
        roi_y = self.lineEdit_13.text()
        roi_w = self.lineEdit_14.text()
        roi_h = self.lineEdit_15.text()
        roi_d = self.lineEdit_16.text()

        # 模型设置
        model_path = self.lineEdit.text()
        image_x = self.lineEdit_2.text()
        image_y = self.lineEdit_3.text()
        conf_threshold = self.lineEdit_4.text()
        nms_iou_threshold = self.lineEdit_5.text()
        try:
            hardware_select = self.hardware_select.checkedButton().text()
        except:
            QMessageBox.warning(self, '警告', '硬件必须选一个', QMessageBox.Yes | QMessageBox.No)
            return

        # 通讯设置
        try:
            output_select = self.output_select.checkedButton().text()
        except:
            QMessageBox.warning(self, '警告', '输出格式必须选一个', QMessageBox.Yes | QMessageBox.No)
            return
        serv_ip = self.lineEdit_6.text()
        serv_post = self.lineEdit_7.text()

        return [[trige_select, camera_ip, data_distance, saveimg, savepcl], [roi_x, roi_y, roi_w, roi_h, roi_d],
                [model_path, image_x, image_y,
                 conf_threshold, nms_iou_threshold, hardware_select], [output_select, serv_ip, serv_post]]

    def start_det(self):
        # [trigger_mode,data_mode,saveimg,savepcl,roi_x,roi_y,roi_w,roi_h,roi_d,image_x,image_y,conf_threshold,nms_iou_threshold]
        result = self.get_data()
        if result is None:
            return
        try:
            # 采集数据检测
            if result[0][0] == "触发":
                self.display(result)
            else:
                if self.global_button_status:
                    self.pushButton_4.setText("结束")
                    self.global_button_status = False
                    # 利用新的线程开启视频流
                    self.t = threading.Thread(target=self.thread_display, args=(result,))
                    self.t.start()
                else:
                    self.pushButton_4.setText("开始")
                    self.global_button_status = True
                    stop_thread(self.t)
                    self.vit.close()
        except:
            pass

    def thread_display(self, result):
        while 1:
            self.display(result)

    def updata_textbrowser1(self,text):
        self.textBrowser.append(text)

    def updata_textbrowser2(self,text):
        self.textBrowser_2.append(text)
    # np.array([[trige_select, camera_ip,data_distance, saveimg, savepcl], [roi_x, roi_y, roi_w, roi_h, roi_d], [model_path,image_x, image_y,
    #                          conf_threshold, nms_iou_threshold,hardware_select],[output_select,serv_ip,serv_post]])
    def display(self, result):
        try:
            t1 = time.time()
            myData = self.vit.get_single_data()
            intensityData = myData.depthmap.intensity
            distanceData = myData.depthmap.distance
            numCols = myData.cameraParams.width
            numRows = myData.cameraParams.height
            distanceData = np.reshape(distanceData, (numRows, numCols)).astype(int)
            if result[1][4] != '':
                distanceData = np.clip(distanceData, 0, int(result[8]))

            if result[0][2]:
                _distancdData = np.copy(distanceData)
                uint8_img = cv2.normalize(src=_distancdData, dst=None, alpha=0, beta=255,
                                          norm_type=cv2.NORM_MINMAX,
                                          dtype=cv2.CV_8U)
                img_ = cv2.cvtColor(uint8_img, cv2.COLOR_GRAY2BGR)
                im_color = cv2.applyColorMap(cv2.convertScaleAbs(img_, alpha=1), cv2.COLORMAP_JET)
                # 显示

                showImage = QImage(im_color.data, im_color.shape[1], im_color.shape[0], QImage.Format_RGB888)
                # image2 = QPixmap(showImage).scaled(self.label_2.width(), self.label_2.height())
                # self.label_2.setPixmap(image2)
                self.label_2.setPixmap(QPixmap.fromImage(showImage))

            intensityDataArray = np.uint16(np.reshape(intensityData, (numRows, numCols)))
            uint16_img = np.clip(intensityDataArray, 0, 255)
            uint8_img = np.uint8(uint16_img)
            im_color = cv2.cvtColor(uint8_img, cv2.COLOR_GRAY2BGR)
            # save img
            if result[0][3]:
                import datetime
                filename = datetime.datetime.now().strftime('%d_%H_%M_%S') + '.jpg'
                cv2.imwrite(filename, im_color)

            # todo save pcl

            # roi 左上角  w  h
            if result[1][0] != '' and result[1][1] != '' and result[1][2] != '' and result[1][3] != '':
                roi_img = np.zeros_like(im_color)
                x = int(result[1][0])
                y = int(result[1][1])
                w = int(result[1][2])
                h = int(result[1][3])
                roi_img[y:y + h, x:x + w, :] = im_color[y:y + h, x:x + w, :]
            else:
                roi_img = im_color

            r = 1
            # todo resize
            if result[2][1] != '' and result[2][2] != '':
                shape = roi_img.shape[:2]  # current shape [height, width]

                new_shape = (int(result[2][1]), int(result[2][2]))

                # Scale ratio (new / old)
                r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

                new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

                if shape[::-1] != new_unpad:  # resize
                    roi_img = cv2.resize(roi_img, new_unpad, interpolation=cv2.INTER_LINEAR)

            # 推理

            if result[2][3] != '':
                result_box, vis_im = self.models.predict(roi_img, conf_threshold=result[2][3])
            else:
                result_box, vis_im = self.models.predict(roi_img)



            if len(result_box.boxes) != 0:
                # 数据处理
                process_result, vis_im = self.process_data(result_box, distanceData, myData, r, vis_im)

                # 数据展示
                if len(process_result)!=0:
                    import csv
                    fo = open("info.csv", "a", newline='')
                    writer = csv.writer(fo)
                    # todo 输出多个或者单个
                    writer.writerows(process_result)
                    # 关闭文件
                    fo.close()
                    if result[3][0] == '多个':
                        res = ','.join(process_result.astype(str).ravel())
                    else:
                        res = ','.join(process_result[0].astype(str).ravel())
                    if len(self.client_socket_list):
                        self.dialogSignel2.emit(res)
                        packet = b'star'  # Start sequence
                        packet += struct.pack('!i', 0)  # Command ID
                        packet += struct.pack('!i', 0)  # Error code (unused)
                        packet += struct.pack('!i', 62)  # Payload length (in bytes)
                        # time  confi  x y z lx ly lz rx ry rz  roll pich  yaw
                        packet += pack_data(0, 0, process_result[0][0], process_result[0][1], process_result[0][2], 0,0,0,0,0,0,process_result[0][3], process_result[0][4], process_result[0][5])  # Payload data
                        packet += b'stop\r\n'  # Stop sequence
                        self.tcp_send(packet)
                    self.dialogSignel1.emit('x,y,z,rotx,roty,rotz,θ=' + res)
                    self.lineEdit_9.setText(str(len(process_result)))
                else:
                    self.lineEdit_9.setText('0')
                    packet = b'stop\r\n'  # Stop sequence
                    self.tcp_send(packet)
            else:
                self.lineEdit_9.setText('0')
                packet = b'stop\r\n'  # Stop sequence
                self.tcp_send(packet)

            # 显示
            showImage = QImage(vis_im.data, vis_im.shape[1], vis_im.shape[0], QImage.Format_RGB888)
            # image2 = QPixmap(showImage).scaled(self.label_12.width(), self.label_12.height())
            # self.label.setPixmap(image2)
            self.label.setPixmap(QPixmap.fromImage(showImage))
            self.lineEdit_8.setText('%.4s ms' % ((time.time() - t1) * 1000))
        except Exception as e:
            self.dialogSignel1.emit(str(e))

    def process_data(self, result_box, distanceData, myData, r, vis_im):
        # 取3个点求平面的法向量和角度
        # 取两个box的两个边的中心点
        try:

            boxes = np.array(result_box.boxes)
            boxes /= r
            scores = result_box.scores
            labels = result_box.label_ids
            # 单个栈板
            pallets = []
            # 2个栈孔
            holes = []
            for i, label in enumerate(labels):
                if label == 1:
                    pallets.append(boxes[i])
                else:
                    holes.append(boxes[i])
            # 把2个孔洞和单个栈板合成一个list
            all_data = []
            for pallet in pallets:
                _tem = []
                _tem.append(pallet)
                for hole in holes:  # xmin  ymin  xmax ymax
                    if hole[0] > pallet[0] and hole[2] < pallet[2] and hole[1] > pallet[1] and hole[3] < pallet[3]:
                        _tem.append(hole)
                all_data.append(_tem)

            # 过滤数据并求得3个点 包括中心点
            result = []
            for data in all_data:
                if len(data) == 3:  # data[1:2] 2个孔洞 两个边的中心点 和两侧的中心点
                    two_hole = data[1:3]
                    two_hole.sort(key=lambda x: x[0])
                    lx = two_hole[0][2]
                    ly = two_hole[0][3]
                    rx = two_hole[1][0]
                    ry = two_hole[1][1]
                    center_x = (lx + rx) / 2
                    center_y = (ly + ry) / 2
                    # 根据xy  获取高度
                    center_z = distanceData[int(center_y), int(center_x)]
                    w = two_hole[0][2] - two_hole[0][0]
                    h = two_hole[0][3] - two_hole[0][1]
                    if center_z == 0 or w <= 0 or h <= 0:
                        continue
                    # # point1
                    # point1 = []
                    # point1_cx = two_hole[0][0]
                    # point1_cy = two_hole[0][1] + h / 2
                    # for i in range(int(point1_cx - w / 4), int(point1_cx - w / 5)):
                    #     for j in range(int(point1_cy - h / 10), int(point1_cy + h / 10)):
                    #         point1_cz = distanceData[int(j), int(i)]
                    #         if point1_cz == 0:
                    #             continue
                    #         else:
                    #             point1.append([i, j, point1_cz])
                    #
                    #
                    # if distanceData[int(point1_cy), int(point1_cx-15)]!=0:
                    #     point1.append([point1_cx-15,point1_cy,distanceData[int(point1_cy), int(point1_cx-15)]])
                    #
                    # # point2
                    # point2=[]
                    # point2_cx = two_hole[1][2]
                    # point2_cy = two_hole[1][3] - h / 2
                    # for i in range(int(point2_cx + w / 5), int(point2_cx + w / 4)):
                    #     for j in range(int(point2_cy - h / 10), int(point2_cy + h / 10)):
                    #         point2_cz = distanceData[int(j), int(i)]
                    #         if point2_cz == 0:
                    #             continue
                    #         else:
                    #             point2.append([i, j, point2_cz])
                    # # print(point2)
                    # if distanceData[int(point2_cy), int(point2_cx + 15)]!=0:
                    #     point2.append([point2_cx + 15,point2_cy,distanceData[int(point2_cy), int(point2_cx + 15)]])
                    all_point=[]
                    all_point.append([center_x, center_y, center_z])
                    h=(data[0][3]-data[0][1])/5
                    for i in range(int(data[0][0]+2), int(data[0][2]-2)):
                        for j in range(int(data[0][1]+h), int(data[0][3]-h)):
                            point1_cz = distanceData[int(j), int(i)]
                            if point1_cz == 0:
                                continue
                            else:
                                if i>two_hole[0][0]-2 and j> two_hole[0][1]-2 and i<two_hole[0][2]+2 and j< two_hole[0][3]+2:
                                    continue
                                if i>two_hole[1][0]-2 and j> two_hole[1][1]-2 and i<two_hole[1][2]+2 and j< two_hole[1][3]+2:
                                    continue
                                all_point.append([i, j, point1_cz])
                    if len(all_point)==0:
                        continue
                    # points_list=np.vstack((np.array(point1),np.array(point2)))
                    # points_list=np.vstack((np.array([center_x, center_y, center_z]),points_list))
                    points_list=np.array(all_point)
                    drow_poinst=points_list[:,:2].copy()

                    # 相机内参转换
                    cam_points = self.vit.new_cam2word(points_list, myData)
                    # todo  角度计算
                    # angle = point_angle(cam_points)
                    # angle=fit_face(cam_points)
                    angle, inliers = fit(cam_points)
                    # angle,inliers=open3d_segment(cam_points)
                    result.append(np.hstack((cam_points[0], angle)))
                    # 画3个点
                    drow_poinst = (drow_poinst * r).astype(np.uint16)
                    for point in drow_poinst[inliers,:]:
                        cv2.circle(vis_im, tuple(point), 1, (255, 0, 0), 1)
            # 保留两位小数
            if len(result)!=0:
                result = np.around(result, decimals=2)
                # 按照高度排序
                index = np.lexsort([result[:, 2]])
                result = result[index, :]
                return np.array(result), vis_im
            else:
                return [], vis_im
        except Exception as e:
            self.dialogSignel1.emit(str(e))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(palette=LightPalette, qt_api='pyqt5'))

    # 设置启动界面
    splash = MySplashScreen()
    # 初始图片
    splash.setPixmap(QPixmap('SICK-logo.png'))  # 设置背景图片
    # 初始文本
    splash.showMessage("正在加载...... ", QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom, QtCore.Qt.black)
    # 设置字体
    splash.setFont(QFont('微软雅黑', 10))
    # 显示启动界面
    splash.show()
    app.processEvents()  # 处理主进程事件
    window = MainWindow()
    window.show()
    splash.finish(window)  # 隐藏启动界面
    splash.deleteLater()
    sys.exit(app.exec_())
