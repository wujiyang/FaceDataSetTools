#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 21:37:39 2018

@author: wujiyang
"""

import sys 
sys.path.append('/home/wujiyang/caffe/python')
import caffe
import numpy as np
import matplotlib.pyplot as plt


'''
show the feature map of some layer when testing
'''

caffe.set_mode_gpu
caffe.set_device(0)


# load model
deploy_model = './FaceModels/SpherefaceNet-20/sphereface_deploy_20.prototxt'
caffe_model = './FaceModels/SpherefaceNet-20/sphereface_model_20.caffemodel'
net = caffe.Net(deploy_model, caffe_model, caffe.TEST)

# image　preprocess
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  # set the input data blob shape
transformer.set_transpose('data', (2,0,1))                                  # move image channels to outermost dimension
#transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))           # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)                                      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))                               # swap channels from RGB to BGR

im=caffe.io.load_image('test.jpg') # load image
net.blobs['data'].data[...] = transformer.preprocess('data',im)      # data preprocess  


# net forward
output = net.forward()


# print 
print net.blobs['data'].data.shape

print net.blobs['res1_3'].data.shape

# plot layer res1_3 feature map, which have 64 channles
figure = plt.figure()
for i in range(64):
    plt.subplot(8, 8, i+1)
    plt.imshow(net.blobs['res1_3'].data[0,i,:,:])
    plt.axis('off')
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.3, wspace=0.3)
    plt.suptitle('res1_3')
	
plt.savefig('./res1_3.png', dpi=100)
plt.show()

