import albumentations as A
import cv2
import matplotlib
import matplotlib.pyplot as plt
import os
import gc
data_path = '/Users/umang/umang/3-1/Image_Processing/project/pro_fold_test copy/pro_fold_test copy'
final_path = '/Users/umang/umang/3-1/Image_Processing/project/augmented_test'
files_big = []
gc.enable()
matplotlib.use('agg')
def visualize(image, i ,j, name, dirs):
    
    a = os.path.isfile(f'{final_path}/{dirs}/{name}_augmented_{i}_{j}.png')
    if(a==False):
        ax = plt.axes()
        ax.axis('off')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        plt.imshow(image)
        name = name[:-4]
        
    
        plt.savefig(f'{final_path}/{dirs}/{name}_augmented_{i}_{j}.png', bbox_inches='tight', transparent=True, pad_inches=0)
        ax.clear()
        plt.clf()
        plt.close('all')
        del ax
        del image
        del i
        del j
        del dirs
        del name
        del a
    gc.collect()
def augment(image_inc, name, dirs):
    x = [-200, 0, 200]  #
    y = [-150, 0, 80]
    for i in range(3):
        for j in range(3):
            x_val = x[i]
            y_val = y[j]
            transform = A.Compose(
        [A.augmentations.geometric.Affine(
                translate_px={"x": x_val, "y": y_val},
                always_apply=True)]
                )
            if(image_inc!='/Users/umang/umang/3-1/Image_Processing/project/pro_fold_test copy/0/.DS_Store'):
                file_type=image_inc[-4:]
    
                if(file_type==".png"):
    
                    
                    img = cv2.imread(image_inc)
                    augmented_image = transform(image=img)['image']
                    visualize(augmented_image, i, j, name, dirs)
                    cv2.destroyAllWindows()
                    del img
                    del transform
                    del augmented_image
                    
                    gc.collect()
                    
                
def check_image(files):
    file_type=files[-4:]
    
    if(file_type==".png"):
    
        img = cv2.imread(files)
        transform = A.Compose(
            [A.augmentations.geometric.Affine(
                    always_apply=True)]
                    )
        print(f"this is okay {files}")
        aug = transform(image =img)['image']
    




if __name__ == "__main__":
    for dirs in sorted(os.listdir(data_path)):
        
        if(dirs!='.DS_Store'):
            a = os.listdir(f"{data_path}/{dirs}")
            
            if(a!= '.DS_Store'):
                files = a
                
                files_big.append(a) 
   
    for dirs in range(len(files_big)):
        if(os.path.exists(f'augmented_test/{dirs}')):
            pass
        else:
            os.makedirs(f'augmented_test/{dirs}')
    for dirs in range(len(files_big)):
        for files in files_big[dirs]:
            augment(f'{data_path}/{dirs}/{files}',files,dirs)
            print('done')
            #check_image(f'{data_path}/{dirs}/{files}')
            
        
            gc.collect()
            
                


    