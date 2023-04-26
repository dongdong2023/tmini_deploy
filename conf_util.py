# coding=utf-8

'''
Author:Don
date:2022/7/11 18:04
desc:
'''
from ruamel import yaml
file_path='config.yaml'

def read_cof():
	with open(file_path,'r' ,encoding='utf-8') as f:
		data=f.read()
		return yaml.load(data, Loader=yaml.RoundTripLoader)



def write_cof(type,item,data):
	file_cfg=read_cof()
	file_cfg[type][item] = data
	with open(file_path,'w',encoding='utf-8') as f:
		yaml.dump(file_cfg,f,Dumper=yaml.RoundTripDumper)





