# encoding=utf-8
import common.data_io.SsrLoader as ssrloader
from common.Control import Control
from common.Streaming import Data
from common.Stream import Streaming
from common.Streaming.BlobServerConfiguration import BlobClientConfig
import cv2
import numpy as np


class VITMINI:

    def __init__(self, ip, control_port=2122, streaming_port=2114, protocol="Cola2"):
        self.ip = ip
        self.control_port = control_port
        self.streaming_port = streaming_port
        self.protocol = protocol

    def init_camera(self):
        # create and open a control connection to the device
        self.deviceControl = Control(self.ip, self.protocol, self.control_port)
        self.deviceControl.open()
        self.deviceControl.singleStep()
        # the device starts stream automatically int the init process
        # deviceControl.initStream()
        # # stop streaming
        # deviceControl.stopStream()
        self.streaming_device = Streaming(self.ip, self.streaming_port)

    def load_ssr(self, path):
        images_distance, images_intensity, images_confidence, cam_params, is_stereo = ssrloader.readSsrData(path, 0, 0)
        numRows = cam_params.height
        numCols = cam_params.width
        for i in range(len(images_intensity)):
            # re=images_intensity[i]
            # intensityDataArray = np.clip(images_intensity[i], 0, 2000)
            # intensityDataArray = np.uint16(np.reshape(intensityDataArray, (numRows, numCols)))
            uint8_img = cv2.normalize(src=images_intensity[i], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                      dtype=cv2.CV_8U)
            # uint16_img = intensityDataArray / (intensityDataArray.max() - intensityDataArray.min())
            # uint16_img *= 255
            # uint8_img = np.uint8(uint16_img)
            # cv2.imshow('img', uint8_img)
            # cv2.waitKey(0)
            uint8_img = cv2.cvtColor(uint8_img, cv2.COLOR_GRAY2BGR)
            res_boxs, mosaic = self.detect(uint8_img)
            cv2.imshow('img', mosaic)
            cv2.waitKey(100)

    def get_continue_data(self):
        try:
            while True:
                self.streaming_device.getFrame()
                wholeFrame = self.streaming_device.frame
                self.myData.read(wholeFrame)
                if self.myData.hasDepthMap:
                    intensityData = self.myData.depthmap.intensity
                    distanceData = self.myData.depthmap.distance
                    numCols = self.myData.cameraParams.width
                    numRows = self.myData.cameraParams.height
                    # # res=(res-1000)/1000*255
                    # intensityDataArray = np.uint16(np.reshape(intensityData, (numRows, numCols)))
                    # # img_ = cv2.normalize(src=intensityDataArray, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                    # #                      dtype=cv2.CV_8U)
                    # # im_color = cv2.applyColorMap(cv2.convertScaleAbs(img_, alpha=1), cv2.COLORMAP_JET)
                    # # intensityDataArray = np.uint16(np.reshape(intensityData, (numRows, numCols)))
                    # uint16_img = intensityDataArray / (intensityDataArray.max() - intensityDataArray.min())
                    # uint16_img *= 255
                    # uint8_img = np.uint8(uint16_img)
                    # im_color=cv2.cvtColor(uint8_img, cv2.COLOR_GRAY2BGR)
                    return intensityData,distanceData
                            # q.put(uint8_img)
                        #     res_boxs, mosaic = self.detect(uint8_img)
                        #     distanceDataArray = np.uint16(np.reshape(distanceData[i], (numRows, numCols)))
                        #
                        #     box_xywh = xyxy2xywh(res_boxs)
                        #     centerx1 = box_xywh[0, 0]
                        #     centerx2 = box_xywh[1, 0]
                        #     centery1 = box_xywh[0, 1]
                        #     centery2 = box_xywh[1, 1]
                        #     z1 = distanceDataArray[centerx1][centery1]
                        #     z2 = distanceDataArray[centerx2][centery2]
                        #     cam_center1 = self.cam2word(centerx1, centery1, z1)
                        #     cam_center2 = self.cam2word(centerx2, centery2, z2)
                        #     angle = np.arctan(np.abs(cam_center1[0] - cam_center2[0]) / np.abs(z1 - z2))
                        # print([cam_center1, cam_center2, angle])
                        # # todo  数据组合
                        # self.tcp_send()
        except KeyboardInterrupt:
            print("")
            print("Terminating")
        except Exception as e:
            print(f"Exception -{e.args[0]}- occurred, check your device configuration")

    def get_single_data(self):
        try:
            self.streaming_device.openStream()

            # request the whole frame data
            self.streaming_device.getFrame()

            # access the new frame via the corresponding class attribute
            wholeFrame = self.streaming_device.frame

            # create new Data object to hold/parse/handle frame data
            myData = Data.Data()
            # parse frame data to Data object for further data processing
            myData.read(wholeFrame)

            if myData.hasDepthMap:
                self.streaming_device.closeStream()
                # self.deviceControl.startStream()
                return myData
        except Exception as e:
            self.streaming_device.closeStream()
            # self.deviceControl.startStream()
            # self.deviceControl.initStream()
            print(e)

    def close(self):
        self.streaming_device.closeStream()
        self.deviceControl.startStream()
    #  todo  传进来矩阵[[x,y,z],[x,y,z]]
    def cam2word(self,box_list,myData):
        stereo=myData.xmlParser.stereo
        box_list=np.array(box_list)
        cx = myData.cameraParams.cx
        fx = myData.cameraParams.fx
        cy = myData.cameraParams.cy
        fy = myData.cameraParams.fy
        m_c2w = myData.cameraParams.cam2worldMatrix
        xp = (cx - box_list[:,0]) / fx
        yp = (cy - box_list[:,1]) / fy
        xc = xp * box_list[:,2]
        yc = yp * box_list[:,2]
        box_list[:,0] = m_c2w[3] +box_list[:,2] *m_c2w[2] +yc *m_c2w[1] +xc *m_c2w[0]
        box_list[:,1] = m_c2w[7] +box_list[:,2] *m_c2w[6] +yc *m_c2w[5] +xc *m_c2w[4]
        box_list[:,2] = m_c2w[11] +box_list[:,2] *m_c2w[10] +yc *m_c2w[9] +xc *m_c2w[8]

        return box_list

    def new_cam2word(self,box_list,myData):
        cx = myData.cameraParams.cx
        fx = myData.cameraParams.fx
        cy = myData.cameraParams.cy
        fy = myData.cameraParams.fy
        m_c2w = myData.cameraParams.cam2worldMatrix
        xp = (cx - box_list[:,0]) / fx
        yp = (cy - box_list[:,1]) / fy

        r2 = (xp * xp + yp * yp)
        r4 = r2 * r2

        k = 1 + myData.cameraParams.k1 * r2 + myData.cameraParams.k2 * r4

        xd = xp * k
        yd = yp * k

        d = box_list[:,2]
        s0 = np.sqrt(xd * xd + yd * yd + 1)

        xc = xd * d / s0
        yc = yd * d / s0
        zc = d / s0 - myData.cameraParams.f2rc

        box_list[:,0] = m_c2w[3] +zc *m_c2w[2] +yc *m_c2w[1] +xc *m_c2w[0]
        box_list[:,1] = m_c2w[7] +zc *m_c2w[6] +yc *m_c2w[5] +xc *m_c2w[4]
        box_list[:,2] = m_c2w[11] +zc*m_c2w[10] +yc *m_c2w[9] +xc *m_c2w[8]

        return box_list
