import os
from os import path
import random
import shutil
path_fold = '/Users/umang/umang/3-1/Image_Processing/project/model/pytorch-image-classification/eval_fold_aug'
if(path.exists(path_fold)):
    pass
else:
  os.mkdir(path_fold)
  pass
data_path = '/Users/umang/umang/3-1/Image_Processing/project/augmented'
files_big = []

first_file = []

last_file = []


for dirs in sorted(os.listdir(data_path)):
    files = []
   
    if(dirs!='.DS_Store'):
       
        
        a = sorted(os.listdir(f"{data_path}/{dirs}"))
        
        if(a!= '.DS_Store'):
            files = a
        first_file.append(files[0])
        last_file.append(files[-1])
        files_big.append(files)



first_file_name =[]

last_file_name = []
def picK_rand():
    a= random.randint(0,2)
    b = random.randint(0,2)
    return a, b


for dirs in range(len(files_big)):
    first = files_big[dirs]
    len_first = len(first)
    final_path = f'{path_fold}/{dirs}'
    if(path.exists(final_path)):
        pass
    else:
        os.makedirs(final_path)
    ind = (first_file[dirs]).find('_')
    name = first_file[dirs][0:ind]
    first_file_name.append(name)
    #print(first_file_name[dirs])
    
    ind = (last_file[dirs]).find('_')
    name = last_file[dirs][0:ind]
    last_file_name.append(name)
    #print(first_file_name)
    

numbers = 50
for dirs in range(len(files_big)):
    if dirs ==0:
        numbers=200
    elif dirs ==1: 
        numbers = 200
    elif dirs ==2:
        numbers = 200
    elif dirs ==3: 
        numbers = 200
    elif dirs == 4:
        numbers = 200
    
    for i in range(numbers):
        if(dirs!=4):
            num = random.randint(int(first_file_name[dirs]), int(first_file_name[dirs+1])-1)
        else:
            num = random.randint(int(first_file_name[dirs]), 87552)
        b, c = picK_rand()
        #print(num)
        
        if(os.path.isfile(f'{data_path}/{dirs}/{num}_{dirs}_augmented_{b}_{c}.png')):
            print('hi')
            shutil.copy(f'{data_path}/{dirs}/{num}_{dirs}_augmented_{b}_{c}.png', f'{path_fold}/{dirs}/{num}_{dirs}_augmented_{b}_{c}.png')
            os.remove(f'{data_path}/{dirs}/{num}_{dirs}_augmented_{b}_{c}.png')

