from skimage import io
from scipy import ndimage
import scipy.ndimage
import sys
import os
import re
import matplotlib.pyplot as plt

from osgeo import gdal
import numpy as np

#function to import Sential-2 L1C Products. For cloud removal algorithm, function load only 10m and 20m resolution bands(2,3,4,5,6,7,8,8A,11,12)
def import_data(img_num,img_path=None):
	path=[[[]for j in xrange(10)] for i in xrange(img_num)] #creating a path list to keep path of the all images
	if img_path is None:
		img_path = ''
	dic_num =1
	bands = ['02','03','04','05','06','07','08','8A','11','12'] #bands with 10mand 20m resolution
	#loops to finding all images path
	for i in range(img_num):
		tem_pat = img_path + ("{}/IMG_DATA/".format(dic_num))
		for j in range(10):
			for f in os.listdir(tem_pat):
				if re.search(r'[a-zA-Z0-9]*({})\.jp2$'.format(bands[j]),f):
					path[i][j] = tem_pat+f
		dic_num+=1
	image_values,image_name,a,b,band_name = [],[],[],[],[]
	#loading images to list 'b' (shape(5,10,1000,1000)) and band names to list 'a' (shape(5,10,1000,1000))
	for j in range(img_num):
		for filename in path[j]:
			#im = gdal.Open(filename)
			im = io.imread(filename, plugin="freeimage")
			#arr = im.ReadAsArray()
			band_name.append(filename)
			image_values.append(im)
		a.append(band_name)
		b.append(image_values)
		band_name=[]
		image_values =[]
	return a,b	#return function to band name list 'a' and images list 'b'
#importing tci images to visualize it
def import_tci(img_num,img_path=None):
	path=[]
        if img_path is None:
                img_path = ''
        dic_num =1
	for i in range(img_num):
                tem_pat = img_path + ("{}/IMG_DATA/".format(dic_num))
		path.append([tem_pat+f for f in os.listdir(tem_pat) if re.search(r'[a-zA-Z0-9]*(TCI)\.jp2$',f)])
		dic_num+=1
	image_values,image_name,a,b,band_name = [],[],[],[],[]
        for j in range(img_num):
		#im = gdal.Open(filename)
                im = io.imread(path[j][0], plugin="freeimage")
                #arr = im.ReadAsArray()
                band_name.append(path[j][0])
                image_values.append(im)
	return band_name,image_values

#resemple funtion to resample bands, which have 20m resolution ,to 10m resolution
def resample(a,b,zoom_size,ord):
	#ord => resampling method (0 = nearest,1 = bilinear...)
	#a => band names list
	#b => images list
	resampled_data=[[[]for l in xrange(len(a[0]))] for k in xrange(len(a))]
	names_data = [[[]for l in xrange(len(a[0]))] for k in xrange(len(a))]
	twen_meter_res = ['02','03','04','08']
	for i  in range(len(a)):
		for j in range(len(a[i])):
			tem_str = a[i][j][-6:-4]
			if tem_str in twen_meter_res:
				resampled_data[i][j] = scipy.ndimage.zoom(b[i][j],zoom_size,order=ord) 
				names_data[i][j] = a[i][j]
			else:
				resampled_data[i][j]=b[i][j]
				names_data[i][j] = a[i][j]
	return names_data,resampled_data #return resampled images list 'resampled data', and names of images 'names_data'

#crop function to crop patch all images
def crop_box(a,b,size,left_corner):
	#for around DLR left_corner = [6550,6700]
	#size ==> patch size 
        x = left_corner[1]
        y = left_corner[0]
        cropped= np.zeros(shape=(len(b),len(b[0]),size,size),dtype='uint16')
        for i in range(len(b)):
                for j in range(len(b[i])):
                        cropped[i][j]=b[i][j][x:x+size,y:y+size]
        return cropped #return cropped images list 'cropped'

def crop_mask(mask,size,left_corner):
	x = left_corner[1]
        y = left_corner[0]
	cropped= mask[x:x+size,y:y+size]
	return cropped

def statistics_all(band_num,a,b):
	for i in range(len(a)):
		for j in range(len(a[i])):
			if a[i][j][-6:-4] == band_num:
                     		print a[i][j]
				print  '  min' ,'   ', ' max  ','   ', '  mean', '   ', '              std'
                     		print  '  ',b[i][j].min(),'    ',b[i][j].max(),'    ' ,b[i][j].mean(),'     ' ,b[i][j].std() 
                    		break

#visualize images, 
def visualise(cropp,columns,rows):
	#cropp==> image list  
	for i in range(len(cropp)):
		fig=plt.figure(figsize=(8, 8))
		for j in range(1,columns*rows +1):
    			fig.add_subplot(rows, columns, j)
			k = ((cropp[i][j-1].astype('float')/cropp[i][j-1].max())*65536).astype('uint16')
    			plt.imshow(k,cmap='gray')
	plt.show()

def visualise_tci(tci):
	for i in range(len(tci)):
		fig=plt.figure(figsize=(8, 8))
		plt.imshow(tci[i],cmap='gray')
	plt.show()

def save_as_envi(mask_path=None):
	if mask_path is None:
                mask_path = ''
	temp_path = mask_path + 'MSK_CLOUDS_B00.gml'
	output = 'outputs/cloud_mask.bsq'
	cmd = "gdal_rasterize -of ENVI -burn 1 -ot Byte -a_srs EPSG:32632 -tr 10 10 MSK_CLOUDS_B00.gml outputs/cloud_mask.bsq"
	os.system(cmd)

def save_as_binary(name_data,res_data):
	for i in range(1,len(name_data)+1):
		dir_name = 'outputs_20m/{}'.format(i)
		if not os.path.exists(dir_name):
			os.makedirs(dir_name)
	for i in range(len(name_data)):
		dir_save = 'outputs_20m/{}/{}.bsq'.format(name_data[i][0][0],name_data[i][0][18:26])
		res_data[i].tofile(dir_save)

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

def load_clo_mask():
	d = np.fromfile('outputs/cloud_mask.bsq',dtype= 'uint8').reshape(10980,10980)
	return d



def apply_mask(img,mask,inde):
	mask[mask==0],mask[mask==1]=2,0
	mask[mask==2]=1
	for i in range(len(img[0])):
		img[inde][i] = img[inde][i]*mask
	return img

















