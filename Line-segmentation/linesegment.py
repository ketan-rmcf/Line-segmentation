import os
from os import path
import pdb
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt 
import argparse
import sys


def createKernel(kernelSize=25, sigma=11, theta=7):
	"""create anisotropic filter kernel according to given parameters"""
	assert kernelSize % 2 # must be odd size
	halfSize = kernelSize // 2
	
	kernel = np.zeros([kernelSize, kernelSize])
	sigmaX = sigma
	sigmaY = sigma * theta
	
	for i in range(kernelSize):
		for j in range(kernelSize):
			x = i - halfSize
			y = j - halfSize
			
			expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
			xTerm = (x**2 - sigmaX**2) / (2 * math.pi * sigmaX**5 * sigmaY)
			yTerm = (y**2 - sigmaY**2) / (2 * math.pi * sigmaY**5 * sigmaX)
			
			kernel[i, j] = (xTerm + yTerm) * expTerm

	kernel = kernel / np.sum(kernel)
	return kernel

def lineseg(filepath,thresholdratio=0.1,minimumspace=3):
    img=cv2.imread(filepath,0)
    height,width=img.shape
    #Preprocess image. -Morphological Transformation and Thresholding
    kernel=createKernel()
    imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    (_, imgThres) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    #Traverse pixel from top to bottom
    pixelwise=[]
    for ypixel in range(imgThres.shape[0]):
        #print(i)
        pixelvalues=imgThres[ypixel:ypixel+1,:]
        num_zeros = (pixelvalues == 0).sum()
        pixelwise.append(num_zeros)
    
    #Visualise distribution
    #y_pos = np.arange(len(pixelwise))
    #plt.bar(y_pos,pixelwise)
    #plt.show()
    
    #Get y cordinates where frequency of black pixel > threshold
    box_of_entries=[]
    for index,entry in enumerate(pixelwise):
        if entry > thresholdratio * width :
            box_of_entries.append(index)
    
    #Simple logic to combine empty spaces         
    ypixelspace=[]
    for i,value in enumerate(box_of_entries):
        if box_of_entries[i]-box_of_entries[i-1]>minimumspace:
            ypixelspace.append(box_of_entries[i]-2)  

    fname=path.basename(filepath).split('.')[0]
    #Break image into lines 
    cnt=1
    minypoint=0
    maxypoint=img.shape[0]
    image_set=[]
    image_set.append(img[0:ypixelspace[0],0:width])
    cv2.imwrite('output/'+fname+'/'+'0.jpg',img[0:ypixelspace[0],0:width])
    image_set.append(img[ypixelspace[-1]:maxypoint,0:width])
    try:
        os.mkdir('output/'+fname)
    except Exception as e:
        print(str(e))    
    for index in range(len(ypixelspace)-1):
        linepiece=img[ypixelspace[index]:ypixelspace[index+1],0:width]
        image_set.append(img[ypixelspace[index]:ypixelspace[index+1],0:width])
        cv2.imwrite('output/'+fname+'/'+str(cnt)+'.jpg',linepiece)
        cnt=cnt+1
    cv2.imwrite('output/'+fname+'/'+str(cnt)+'.jpg',img[ypixelspace[-1]:maxypoint,0:width])
    
    print(image_set)  
    return path.join(os.getcwd(),'output',fname)    
        
def main():
    parser = argparse.ArgumentParser(description='Line Segmentation of images. ')
    parser.add_argument('file',help="Image path here.")
    args = parser.parse_args()
    if args.file and path.exists(args.file):
        filepath=args.file
        op_folder=lineseg(filepath)
        print("Segmentation succesful!")
        print("Check folder : "+op_folder)
        return 1
    else:
        print("Enter correct image path!!")
        return 0



if __name__=='__main__':
    status=main()
