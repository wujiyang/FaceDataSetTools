import caffe
import numpy as np
import matplotlib.pyplot as plt

caffe.set_device(1)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('./gait.prototxt')
#solver.net.copy_from('kenel256_att_iter_2000.caffemodel')
for j in range(15000):
	solver.step(1)
	if j % 100 == 0:
		for i in range(16):
			plt.subplot(4,4,i+1)
			plt.imshow(solver.net.blobs['conv3'].data[0,i,:,:])
			plt.axis('off')
			plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,hspace=0, wspace=0.05)
			#plt.suptitle('{}'.format(j))
			a="conv3"
			plt.suptitle('{}'.format(a))
		plt.savefig('./tmp/base/{0:5s},{1:5d}.png'.format(a,j), dpi=100)
		plt.close()
		# for i in range(2):
			# plt.subplot(1,2,i+1)
			# plt.imshow(solver.net.blobs['conv2_att_my'].data[0,i,:,:])
			# plt.axis('off')
			# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,hspace=0, wspace=0.05)
			# a="conv2_att_my"
			# plt.suptitle('{}'.format(a))
		# plt.savefig('./tmp/base/{0:12s}{1:5d}.png'.format(a,j), dpi=100)
		# plt.close()
		for i in range(16):
			 plt.subplot(4,4,i+1)
			 plt.imshow(solver.net.blobs['conv2'].data[0,i,:,:])
			 plt.axis('off')
			 plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,hspace=0, wspace=0.05)
			 a="conv2"
			 plt.suptitle('{}'.format(a))
		plt.savefig('./tmp/base/{0:5s},{1:5d}.png'.format(a,j), dpi=100)
		plt.close()
		for i in range(16):
			plt.subplot(4,4,i+1)
			plt.imshow(solver.net.blobs['conv1'].data[0,i,:,:])
			plt.axis('off')
			plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,hspace=0, wspace=0.05)
			a="conv1"
			plt.suptitle('{}'.format(a))
		plt.savefig('./tmp/base/{0:5s},{1:5d}.png'.format(a,j), dpi=100)
		plt.close()
