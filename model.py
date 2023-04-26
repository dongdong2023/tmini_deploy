import fastdeploy as fd
import numpy as np
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    x=np.array(x)
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y
# 配置runtime，加载模型

class Model:

    def init_model(self,path='',use_gpu='cpu'):
        option = fd.RuntimeOption()
        # 切换使用CPU/GPU
        if use_gpu =='' or use_gpu =='CPU':
            option.use_cpu()
        else:
            option.use_gpu()
        # 切换不同后端
        # option.use_paddle_backend() # Paddle Inference
        # option.use_trt_backend() # TensorRT
        # option.use_openvino_backend() # OpenVINO
        # option.use_ort_backend() # ONNX Runtime
        if path=='':
            self.model = fd.vision.detection.YOLOv7('weights/yolov7.onnx', runtime_option=option)
        else:
            self.model = fd.vision.detection.YOLOv7(path, runtime_option=option)


    def predict(self,im,conf_threshold=0.25,nms_iou_threshold=0.5):
        result = self.model.predict(im, conf_threshold=float(conf_threshold), nms_iou_threshold=float(nms_iou_threshold))
        # 预测结果可视化
        vis_im = fd.vision.vis_detection(im, result)

        return  result,vis_im
