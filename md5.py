# encoding=utf-8
import hashlib
from conf_util import read_cof
from camera_utils import VITMINI
cfg = read_cof()
cam_ip = cfg['Camera']['ip']
vit = VITMINI(cam_ip)
vit.init_camera()
data = vit.get_sn()
hash_object = hashlib.md5(data.encode())
print(hash_object.hexdigest())

