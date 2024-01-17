import numpy as np
from PIL import Image, ImageOps
import sys
import random

import cv2
loc = 'pics/0_0.png'
image = Image.open(loc).convert('L')
image.convert('1')
image = ImageOps.invert(image)
image.save('test2.png')

numpy_data = np.array(image)
white_count = 0
black_count = 0
count = 0
rows , coloumns = numpy_data.shape

# numpy_data.setflags(write=1)
# for i in range(rows):
#     for j in range(coloumns):
#         if numpy_data[i][j]== 255:
#             white_count = white_count +1
#         elif numpy_data[i][j]==0:
#             black_count = black_count+1
#         else:
#             count=count+1
#             if(numpy_data[i][j]>125): 
#                 numpy_data[i][j]=255
#             else:
#                 numpy_data[i][j]=0
            
 
data = Image.fromarray(numpy_data) 
      
    #
print(white_count, black_count, count)

data.save('test.png') 


def make_kern(i, j, data, size):
    for a in range(-size, size):
        for b in range(-size, size):
            if(a and b == 0):
                continue            
            if(data[i-a, j-b]<125):
                num = random.randint(1, 3)

                if(num ==2):
                    data[i-a, j-b] = random.randint(1, 125)
                print(data[i-a, j-b])
    return data




size = 2
def add_noise(data):
    rows, columns = data.shape
    
    for i in range(rows-size):
        for j in range(columns-size):
            if(data[i][j]>125):
                data = make_kern(i, j, data, size)
    return data

#numpy_data = add_noise(numpy_data)
#data = Image.fromarray(numpy_data) 
#data.save('test3.png') 
def add_vert(data):
    rows, columns = data.shape
    for i in range(rows-3):
        for j in range(columns-3):
            if(data[i][j])>125:
                num = random.randint(1,4)
                if num==1:
                    
                    
                    data[i][j+1]=255
                    data[i][j]=255
                    
                elif num ==2:
                   
                    data[i][j-1]=255

                    data[i][j]=255
                elif num ==3:
                    
                    
                    data[i+1][j]=255
                    data[i][j]=255
                elif num ==4:
                    
                    
                    data[i-1][j]=255
                    data[i][j]=255
                    
    return data
numpy_data = add_vert(numpy_data)
data = Image.fromarray(numpy_data) 
data.save('test3.png') 



def main():
    
    image = Image.open(loc).convert('L')
    image.convert('1')
    image = ImageOps.invert(image)
    image.save('test2.png')
    numpy_data = np.array(image)
    numpy_data = add_vert(numpy_data)
    data = Image.fromarray(numpy_data) 
    data.save('test3.png') 
if __name__ == "__main__":
    main()





