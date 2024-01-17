import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
import multiprocessing
from sklearn.metrics import confusion_matrix
import sys
import os
from os import path
from PIL import Image
import random
import shutil
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def main():
    # Paths for image directory and model
   # os.mkdir('/Users/umang/umang/3-1/Image_Processing/project/model/pytorch-image-classification/test_images')
  #  path_img = '/Users/umang/umang/3-1/Image_Processing/project/model/pytorch-image-classification/test_images'
    
 #   shutil.copy(image, f"{path_img}/image.png" )
   # image_path = f"{path_img}/image.png"
    
    EVAL_DIR="/Users/umang/umang/3-1/Image_Processing/project/model/pytorch-image-classification/eval_fold"
    EVAL_MODEL='model.pth'
    
 
    if(len(sys.argv)<=1):
        print("Provide an image path")
        exit()
        
    elif(sys.argv[1]=='-r'):
        files = list(os.walk(EVAL_DIR))
        all_files = []
        for i in range(1,6):
       
            for j in range(len(files[i][2])):
                all_files.append(files[i][2][j])
    #print(all_files)
        random_file = random.choice(all_files)
   
        fold = random_file[-5]
        print(f"The random selected file is {random_file}")
        
        image_path =  f"/Users/umang/umang/3-1/Image_Processing/project/model/pytorch-image-classification/eval_fold/{fold}/{random_file}"
    else:
         image_path = sys.argv[1]
  
    # Load the model for evaluation
    model = torch.load(EVAL_MODEL)
    model.eval()

    # Configure batch size and nuber of cpu's
    num_cpu = multiprocessing.cpu_count()
    bs = 1

    # Prepare the eval data loader
    eval_transform=transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
   
    eval_dataset=datasets.ImageFolder(root=EVAL_DIR, transform=eval_transform)
    img = pil_loader(image_path)
   
    transformed_img = eval_transform(img)
    transformed_img = torch.unsqueeze(transformed_img, 0)
    
    #print(transformed_img)
    eval_loader=data.DataLoader(eval_dataset, batch_size=bs, shuffle=True,
                                num_workers=num_cpu, pin_memory=True)

    # Enable gpu mode, if cuda available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Number of classes and dataset-size
    num_classes=len(eval_dataset.classes)
    dsize=len(eval_dataset)

    # Class label names
    class_names=['0', '1', '2', '3', '4']

    # Initialize the prediction and label lists
    predlist=torch.zeros(0,dtype=torch.long, device='cpu')
    lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

    # Evaluate the model accuracy on the dataset
    correct = 0
    total = 0
    with torch.no_grad():
        # for images, labels in eval_loader:
        #     images, labels = images.to(device), labels.to(device)
            outputs = model(transformed_img)
            _, predicted = torch.max(outputs.data, 1)
            
            
            ans = predicted.tolist()
            ans = ans[0]
            
            

            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()

            # predlist=torch.cat([predlist,predicted.view(-1).cpu()])
            # lbllist=torch.cat([lbllist,labels.view(-1).cpu()])
            print(f"The model thinks that the class is {ans}")
    # Overall accuracy
    # overall_accuracy=100 * correct / total
    # print('Accuracy of the network on the {:d} test images: {:.2f}%'.format(dsize,
    #     overall_accuracy))

    # # Confusion matrix
    # conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
    # print('Confusion Matrix')
    # print('-'*16)
    # print(conf_mat,'\n')

    # # Per-class accuracy
    # class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
    # print('Per class accuracy')
    # print('-'*18)
    # for label,accuracy in zip(eval_dataset.classes, class_accuracy):
    #      class_name=class_names[int(label)]
    #      print('Accuracy of class %8s : %0.2f %%'%(class_name, accuracy))

    # '''
    # Sample run: python eval.py eval_ds
    #'''
if __name__ == "__main__":
    main()
