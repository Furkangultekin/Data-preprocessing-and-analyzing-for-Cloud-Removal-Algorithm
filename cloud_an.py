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

def load_data(img_num,img_path=None):
        path=[]
        if img_path is None:
                img_path = ''
        for i in range(img_num):
                tem_pat = img_path + ("outputs/{}/".format(i+1))
                temm = [tem_pat+f for f in os.listdir(tem_pat) if re.search(r'[a-zA-Z0-9]*[l]\.bsq$',f)]
                path.append(temm[0])
        band_name,image_values = [],[]
        for j in range(img_num):
                #im = gdal.Open(filename)
                im = np.fromfile(path[j],dtype = 'uint16').reshape(10,1000,1000)
                #arr = im.ReadAsArray()
                band_name.append(path[j])
                image_values.append(im)
        return band_name,image_values

def select_img(img,inde):
	coun = len(inde)
	sel_im = np.zeros(shape=(coun,10,1000,1000)).astype('uint16')
	for i in range(coun):
		sel_im[i] = img[inde[i]]
	return sel_im

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

def load_tci():
        im = np.fromfile('tci_0626.bsq',dtype = 'uint8').reshape(1024,1024,3)
        return im

def load_clo_mask(typ = None):
	if typ == 'all_area' :
       		d = np.fromfile('outputs/cloud_mask.bsq',dtype= 'uint8').reshape(10980,10980)
        else:
		d = np.fromfile('outputs/cropped_cloud_mask.bsq',dtype= 'uint8').reshape(1000,1000)
	return d

def syn_from_mask(mask):
	synn = np.zeros(shape=(1000,1000)).astype('uint8')
	synn[synn==0]=1
	synn[mask==1]=0
	return synn

def create_syn_center(size,left_corner):
	synthe = np.zeros(shape=(1000,1000)).astype('uint8')
	syn_mask = np.zeros(shape=(1000,1000)).astype('uint8')
	synthe[synthe==0]=1
	x=left_corner[0]
	y=left_corner[1]
	synthe[x:x+size,y:y+size]=0
	syn_mask[synthe==0] = 1
	return synthe,syn_mask

def create_syn_sm_box(size,left_corners):
	syn = np.zeros(shape=(1000,1000)).astype('uint8')
        syn_mas =np.zeros(shape=(1000,1000)).astype('uint8')
	syn[syn==0]=1
        for i in range(len(left_corners)):
		x=left_corners[i][1]
        	y=left_corners[i][0]
        	syn[x:x+size[i],y:y+size[i]]=0
	syn_mas[syn==0] = 1
        return syn,syn_mas

def apply_mask(img,mask,inde):
        for i in range(len(img[0])):
                img[inde][i] = img[inde][i]*mask
        return img

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

def number_of_pixels(im):
        un, coun = np.unique(im,return_counts=True)
        print 'pixel values','--->','number_of_pixel'
        for i in range(len(un)):
                print un[i],'--->',coun[i]

def remove_sat(im,min_thres,max_thres):
        new_img= np.array([[[0 for l in xrange(len(im[0][0]))]for f in xrange(len(im[0]))]for l in xrange(len(im))],dtype='float')
        for i in range(len(new_img)):
                for j in range(len(im[0])):
                        for k in range(len(im[0][0])):
                                if im[i][j][k]<min_thres[i]:
                                        new_img[i][j][k] = min_thres[i]
                                elif im[i][j][k]>max_thres[i]:
                                        new_img[i][j][k] = max_thres[i]
                                else :
                                        new_img[i][j][k] = im[i][j][k]
        return new_img
