#!/usr/bin/env python
# -*-coding:utf8 -*-

'''
合并webface数据集和vggface2数据集
注意相同类别的进行合并
'''
import os
from PIL import Image

def read_overlap(file):
    # 读取overlap文档，输出对应的dict
    simi_dict = {}
    with open(file, 'r') as f:
        alllines = f.readlines()
        for line in alllines:
            line_split = line.strip().split(' ')
            # print line_split[1], line_split[-1]
            simi_dict[line_split[-1]] = line_split[1]

    return simi_dict


def main():
    simi_dict = read_overlap('vggface2_webface_overlap.txt')
    # keys = simi_dict.keys()
    # print keys
    # print len(keys)
    source_folder = '/home/wujiyang/data/CASIA-Webface/webface-aligned-224'
    target_folder = '/home/wujiyang/data/vggface2/vggface2_aligned-224-224'
    dirs = os.listdir(source_folder)
    i = 0
    for item in dirs:
        i = i + 1
        sub_dir = os.path.join(source_folder, item)
        # print i, sub_dir
        files = os.listdir(sub_dir)
        if len(files) < 200:
            continue
        
        if simi_dict.has_key(item):
            print item, 'is existed'
            # 合并
            # for file in files:
            #     img = Image.open(sub_dir + '/' + file)
            #     img.save(target_folder + '/' + simi_dict[item] + '/' + file)
        else:
            print item
            # just copy
            img_dir = os.path.join(target_folder, item)
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)
            for file in files:
                img = Image.open(sub_dir + '/' + file)
                img.save(img_dir + '/' + file)
        

if __name__ == "__main__":
    main()