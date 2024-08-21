import os
import time
import argparse
import torch
import cv2
import models
import pickle
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from PIL import Image 
from bunch import Bunch
from ruamel.yaml import YAML
from torchvision.transforms import Grayscale, Normalize, ToTensor
from utils.helpers import dir_exists, remove_files, double_threshold_iteration
from cv2 import adaptiveThreshold, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY
import ttach as tta
from datatry import ves_data
from utils.helpers import get_instance




def pic_process(data_path, name, patch_size, stride, weight_path,CFG):
    save_path = os.path.join(data_path, f"image_pro")
    dir_exists(save_path)
    remove_files(save_path)
    Image.MAX_IMAGE_PIXELS = None
    img_list = []
    tic=time.time()
    if name == "breast":
        img_path = os.path.join(data_path,  "images")
        file_list = list(sorted(os.listdir(img_path)))
        
        for i, file in enumerate(file_list[:1]):           
            time1=time.time()  
    

            img = Image.open(os.path.join(img_path, file))
            img = Grayscale(1)(img)
            img= ToTensor()(img)

            #get patch
            img = get_patch(img, patch_size, stride)  
            

            #save_patch(img, save_path)

            #crop images --list
            image_list = cut_pic(img, patch_size, stride)
            
            #normalize
            image_list = [normalization(cut_img) for cut_img in image_list] 
            #img = normalization(img)
            #image_list = img
            save_pic(image_list, save_path)

            #img_list.extend(image_list)
            time2=time.time()
            times = time2-time1
            print(f'picture{i} spent {times}s')  

   
    
    # test if vessel exists
    img_list=test_ve( weight_path, save_path, CFG)    
    end_time=time.time()
    total_time=end_time - tic 
    print(f'sum: save picture spent {total_time}s')   


def test_ve( weight_path, save_path, CFG):
    #load model
    model = get_instance(models, 'model', CFG)
    model = nn.DataParallel(model.cuda())  #!!!
    checkpoint = torch.load(weight_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] 
    model.load_state_dict(state_dict) 
    # Process the images in the DataLoader

    test_data = ves_data(save_path)
    test_loader = DataLoader(test_data, 1,
                             shuffle=False,  num_workers=16, pin_memory=True)  
    
    if CFG.tta:
            model = tta.SegmentationTTAWrapper(
                model, tta.aliases.d4_transform(), merge_mode='mean')
    model.eval()

    with torch.no_grad():
    #load img
        for i, sub in enumerate(test_loader):
             #print(sub)
             print(f'pre {i}')
             tic = time.time()
             #imgsave = sub.cpu().numpy()
             #print(imgsave)

             pre = model(sub)
             #presave = pre.cpu().numpy()
             #print(presave)

             H ,W =512, 512
             pre = TF.crop(pre, 0, 0, H, W)
             sub = TF.crop(sub, 0, 0, H, W)
             pre = pre[0,0,...]
             sub = sub[0,0,...]
             predict = torch.sigmoid(pre).cpu().detach().numpy()
             #print(predict.shape)
 
             threshold = CFG.threshold
             predict_b = np.where(predict >= threshold, 1, 0)
             #print(predict_b.shape)
             cv2.imwrite(
                 f"/home/wenqi/RF-UNet/datasets/breast/image_pro/img{i}.png", np.uint8(sub*255))
             cv2.imwrite( 
                  f"/home/wenqi/RF-UNet/datasets/breast/image_pro/pre{i}.png", np.uint8(predict*255))
             if np.any(predict_b == 1):
                 
                 cv2.imwrite(
                       f"/home/wenqi/RF-UNet/datasets/breast/image_pro/pre_b{i}.png", np.uint8(predict_b*255))
             
               

        tic1 = time.time()
        pre_time =tic1 - tic
        print(f'spent {pre_time} s')
                 
             



def normalization(img):
    mean = torch.mean(img)
    std = torch.std(img)
    n = Normalize([mean],  [std])(img)
    n = (n - torch.min(n)) / (torch.max(n) - torch.min(n))
    return n 



def save_patch(imgs_list, path):
    for i, sub in enumerate(imgs_list):
        with open(file=os.path.join(path, f'{i}.pkl'), mode='wb') as file:
            pickle.dump(np.array(sub), file)
            print(f'save : {i}.pkl')


def save_pic(imgs_list, path):
    for i, sub in enumerate(imgs_list):
        with open(file=os.path.join(path, f'pic {i}.pkl'), mode='wb') as file:
            pickle.dump(np.array(sub), file)
            print(sub.shape)
            print(f'save : pic {i}.pkl')


def get_patch(img, patch_size, stride):
    print(img.shape)
    _, h, w = img.shape   #[1, 11483, 18300]
    pad_h = stride - (h - patch_size) % stride if h % patch_size != 0 else 0
    pad_w = stride - (w - patch_size) % stride if w % patch_size != 0 else 0                                                                                                                    
    image = F.pad(img, (0, pad_w, 0, pad_h),"constant", 0)  #sequence order of hw!
    print(f'get_patch')
    print(image.shape)
    return image




def cut_pic(img, patch_size, stride):
    _, h, w = img.shape
    image_list = []
    h_steps = (h - patch_size) // stride + 1
    w_steps = (w - patch_size) // stride + 1
    print(f'h_steps: {h_steps} * w_steps: {w_steps}')
    for i in range(h_steps):
        for j in range(w_steps):
            start_h =max(0, i * stride)
            end_h = min(h, start_h + patch_size) #no exceed
            start_w = max(0, j * stride)
            end_w = min(w, start_w + patch_size)
            #3D   [1:512,512]
            cropped_img = img[:, start_h:end_h, start_w:end_w]
            #print(cropped_img.shape) 
            image_list.append(cropped_img)
    return image_list
