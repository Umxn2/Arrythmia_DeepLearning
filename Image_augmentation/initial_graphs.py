import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os
from os import path
import subprocess
import random

x = np.empty(187)
train_df=pd.read_csv('./mitbih_train.csv',header=None)
test_df=pd.read_csv('./mitbih_test.csv',header=None)
train_df[187]=train_df[187].astype(int)
test_df[187]=test_df[187].astype(int)
folder = "final"
pro_folder = "pro_fold"
file_path =f'{folder}'
# if path.exists(file_path):
#    pass
# else:
#    os.mkdir(folder)


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
se = test_df
for j in range(len(test_df)):
   for i in range(187):
      x[i] = se[i][j]
   ax = plt.axes()
   

   ax.axis('off'),

   

   ax.spines['top'].set_visible(False)
   ax.spines['right'].set_visible(False)
   ax.spines['bottom'].set_visible(False)
   ax.spines['left'].set_visible(False)


   ax.get_xaxis().set_ticks([])
      
   plt.plot(x)
   path_new = f"{se[187][j]}"
   path_final = f"test/{folder}/{se[187][j]}"
   if path.exists(path_final):
       pass
   else:
      os.makedirs(f"test/{folder}/{se[187][j]}")

   plt.savefig(f'test/{folder}/{path_new}/{j}_{se[187][j]}.png')
   loc = f'test/{folder}/{path_new}/{j}_{se[187][j]}.png'
   image = Image.open(loc).convert('L')
   image.convert('1')
   image = ImageOps.invert(image)
   
   numpy_data = np.array(image)
   numpy_data = add_vert(numpy_data)
   data = Image.fromarray(numpy_data) 
   path_final_2 = f"test/{pro_folder}/{se[187][j]}"
   if path.exists(path_final_2): 
       pass
   else:
      os.makedirs(f"test/{pro_folder}/{se[187][j]}")
   data.save(f'test/{pro_folder}/{path_new}/{j}_{se[187][j]}.png') 

   ax.clear()
   plt.clf()

