from skimage import io
from scipy import ndimage
import scipy.ndimage
import sys
import os
import re
import matplotlib.pyplot as plt
import gdalnumeric
from osgeo import gdal
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

def load_rec(dir,file):
        path = []
	dirs = os.listdir(dir)
	for i in dirs: 
		tem_pat = dir+'/'+i+'/'
		temm = [tem_pat+f for f in os.listdir(tem_pat) if f ==file]
                path.append(temm[0])
		temm = None
        band_name,image_values = [],[]
	if file[-7:-4]=='rgb':
		shp = [1000,1000,3]
		ty = 'float'
	elif file[4:9]=='cloud':
		shp=[10,1000,1000]
		ty = 'uint16'
        for j in range(len(path)):
                #im = gdal.Open(filename)
                im = np.fromfile(path[j],dtype = ty).reshape(shp[0],shp[1],shp[2])
                #arr = im.ReadAsArray()
                band_name.append(path[j])
                image_values.append(im)
        return band_name,image_values

def visualise(cropp,columns,rows):
        #cropp==> image list  
        for i in range(len(cropp)):
                fig=plt.figure(figsize=(8, 8))
                for j in range(1,columns*rows +1):
                        fig.add_subplot(rows, columns, j)
                        k = ((cropp[i][j-1].astype('float')/cropp[i][j-1].max())*65536).astype('uint16')
                        plt.imshow(k,cmap='gray')
        plt.show()

def visualise_1(im,columns,rows,color=None):
        fig=plt.figure(figsize=(8,8))
        for i in range(1,columns*rows +1):
                fig.add_subplot(rows,columns,i)
                plt.imshow(im[i-1],cmap=color)
        plt.show()

def normalize(g):
        h=[]
        for i in range(len(g)):
                tem = g[i].astype('float')
                tt = (tem-tem.min())/(tem.max()-tem.min())
                h.append(tt)
        return h

def mk_rgb(bands):
        rgb = np.dstack((bands))
        return rgb

def rms_err(orig,rec):
	rmse=[]
	for i in range(len(orig)):
		rmse.append(sqrt(mean_squared_error(or_0721[i],rec[i])))
	return rmse

def spec_angle(orig,rec,size):
	spec_an = np.zeros(shape=(size*size),dtype='float')
	ori = orig.reshape(10,size*size)
	re = rec.reshape(10,size*size)
	for i in range(len(spec_an)):
		pix_ori = ori[0:10,i].astype('float')
		pix_rec = re[0:10,i].astype('float')
		above= np.sum(np.multiply(pix_ori,pix_rec))
		bel = (np.sqrt(np.sum(np.multiply(pix_ori,pix_ori))))*(np.sqrt(np.sum(np.multiply(pix_rec,pix_rec))))
		spec_an[i] = np.arccos(above/bel)
	return spec_an.reshape(size,size)


def hist_of(im):
	for i in range(len(im)):
		 plt.hist(im[i, :, :].flatten(), bins=100,range=(0,10000), histtype='step', label='original')
	plt.axis([0,10000,0,450000])
	plt.show()

