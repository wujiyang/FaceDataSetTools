#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
@Author: wujiyang
@Time: 2017/09/27 15:30:24
@Brief: a file for general image processing, such as contrast ratio, color to greyscale and so on
'''

import os
import threading
from skimage import data, exposure, img_as_float
from skimage import color, io
import matplotlib.pyplot as plt

def dirList(path, allfiles):
    '''
    递归遍历文件夹，将所有文件名保存到list中
    allfiles为传入的list
    '''
    filelist = os.listdir(path)
    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            dirList(filepath, allfiles)
        else:
            allfiles.append(filepath)


def imgContrastRatio(img):
    '''
    改变图像的对比度和亮度
    '''
    # image = exposure.adjust_gamma(img, 2.5)   #调暗
    # image = exposure.adjust_gamma(img, 0.5) #调亮
    # image = exposure.adjust_log(img, 5)
    image = exposure.adjust_sigmoid(img, 1) 
    return image

def rgb2gray(img):
    '''
    将彩色图像转换成灰度图
    '''
    image = color.rgb2gray(img)
    return image

def saveImage(path, img):
    '''
    将图片保存到指定的路径中
    '''
    io.imsave(path, img)

def findLast(string, str):
    '''
    寻找字符在字符串中最后一次出现的位置，没有的话返回-1
    '''
    lastPositin = -1
    while True:
        position = string.find(str, lastPositin + 1)
        if position == -1:
            return lastPositin
        lastPositin = position
        
    return lastPositin


def img_process(filename, originalDir, targetDir):
    '''
    图像的整个处理过程：读取，变换，保存
    '''
    img = io.imread(filename)
    # img_gray = rgb2gray(img)
    img_contrast = imgContrastRatio(img)
    targetPath = targetDir + filename[filename.index(originalDir) + len(originalDir): ]
    print targetPath
    position = findLast(targetPath, "/")
    if os.path.exists(targetPath[0 : position]) == False:
        os.makedirs(targetPath[0 : position])
    saveImage(targetPath, img_contrast)


def main():
    originalDir = '/home/wujiyang/data/MS-Celeb-10000/aligned_image'
    targetDir = '/home/wujiyang/data/MS-Celeb-10000/aliged_processed'
    allFiles = []
    dirList(originalDir, allFiles)
    for filename in allFiles:
        # 开启多线程处理
        while True:
            if(len(threading.enumerate()) < 100 ):
                break;
        t = threading.Thread(target = img_process, args=(filename, originalDir, targetDir))
        t.start()
    

if __name__ == '__main__':
    main()
