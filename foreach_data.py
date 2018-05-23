#!/usr/bin/env python
# -*-coding:utf8 -*-

import os
import sys
import random

def walk_through_folder_for_split(src_folder):
    val_set = []
    train_set = []
    
    label=0
    for people_folder in sorted(os.listdir(src_folder), key = str.lower):
        people_path = src_folder + people_folder + '/'
        people_imgs = []
        for img_file in sorted(os.listdir(people_path), key = str.lower):
            img_path = people_path+img_file
            people_imgs.append((img_path,label))
	    #if pics of one identity < 25, drop it
        if len(people_imgs) < 25:
             continue
        random.shuffle(people_imgs)

        if len(people_imgs) <= 200:
            val_set += people_imgs[0:5]
            train_set += people_imgs[5:]
        else:
            val_set += people_imgs[0:5]
            train_set += people_imgs[5:]

        sys.stdout.write('\rdone: '+str(label))
        sys.stdout.flush()
        label += 1
    print '\n'
    print 'test set num:%d'%(len(val_set))
    print 'train set num:%d'%(len(train_set))

    return val_set,train_set
def set_to_csv_file(data_set, file_name):
    f = open(file_name,'wb')
    print f
    for item in data_set:
        line = item[0] + ' ' + str(item[1]) + '\n'
        f.write(line)
    f.close()

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print'Usage:python %s src_folder val_set_file train_set_file'%(sys.argv[0])
        sys.exit()
    src_folder = sys.argv[1]
    val_set_file = sys.argv[2]
    train_set_file = sys.argv[3]
    if not src_folder.endswith('/'):
        src_folder += '/'

    val_set,train_set = walk_through_folder_for_split(src_folder)
    set_to_csv_file(val_set, val_set_file)
    set_to_csv_file(train_set, train_set_file)


