#! /usr/bin/env python
# -*- coding:utf-8 -*- 

'''
remove the softmax layer after training process, if possibile, remove the data layer also
only preserve the feature embedding layer
'''

import sys 
sys.path.append('/home/wujiyang/caffe/python')
import caffe
import numpy as np
import matplotlib.pyplot as plt

deploy_model = '../FaceModels/SpherefaceNet-20/sphereface_deploy_20.prototxt'
caffe_model = '../FaceModels/SpherefaceNet-20/sphereface_model_20.caffemodel'
net = caffe.Net(deploy_model, caffe_model, caffe.TEST)  

net.save('../FaceModels/SpherefaceNet-20/final.caffemodel')
