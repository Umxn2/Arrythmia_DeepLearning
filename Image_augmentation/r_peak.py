from PIL import Image
import numpy as np
import cv2
import os
import gc
data_path = '/Users/umang/umang/3-1/Image_Processing/project/pro_fold'
img = '/Users/umang/umang/3-1/Image_Processing/project/pro_fold copy/1/74686_1.png'
img_data = '/Users/umang/umang/3-1/Image_Processing/project/r_peak'
#os.mkdir('img_data')

files_big = []
for dirs in sorted(os.listdir(data_path)):
    files = []
    if(dirs!='.DS_Store'):
        a = os.listdir(f"{data_path}/{dirs}")
        if(a!= '.DS_Store'):
            files = a
            files_big.append(a)
for dirs in range(len(files_big)):
    
    for files in files_big[dirs]:
        pass

def r_peak(img):
    rows, columns = np.shape(img)
 
    for i in range(rows):
        for j in range(columns):
            
            if(img[i][j]>200 and j>150):

                return i , j, rows, columns



def change_val(a,b, img):
    rows, columns = np.shape(img)
    

    for i in range(rows):
        img[i][b] = 255
    return img
def get(img, files, dirs):
    img = Image.open(img)
    numpydata = np.array(img)
    a, b, rows, columns = r_peak(numpydata)
    arr = change_val(a,b,numpydata)

    
        # saving the final output  
        # as a PNG file 
    
  
#maxcolordenoise(img, 50,50,16,75)
    cropped = numpydata[0:,b-100:b+100]
    data = Image.fromarray(cropped) 
    data = data.resize((640, 480), Image.Resampling.LANCZOS)
    data.save(f'{img_data}/{dirs}/{files}_r_peak.png') 


for dirs in range(len(files_big)):
    os.makedirs(f'r_peak/{dirs}')
    for files in files_big[dirs]:
        get(f'{data_path}/{dirs}/{files}',files,dirs)
    gc.collect()

