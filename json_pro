import json
import numpy as np
import cv2
import os
import fnmatch
from pathlib import Path

root_dir = '/home/wenqi/RF-UNet/datasets/json_pro'


def gen_mask_img(json_filename):
    # read json file
    with open(json_filename, "r") as f:
        data = f.read()
    all_points = []
    # convert str to json objs
    data = json.loads(data)
    # get the points   
    target_dirname = os.path.dirname(json_filename)
    original_img_filename = target_dirname + '/' + data["imagePath"]

        # read image to get shape
    image = cv2.imread(original_img_filename)

         # create a blank image
    mask = np.zeros_like(image, dtype=np.uint8)
    print(mask.shape)
    for i in data["shapes"]:
        points = i["points"]
        points = np.array(points, dtype=np.int32).reshape(-1, 2)   # tips: points location must be int32 
        # fill the contour with 255
        cv2.fillPoly(mask, [points], (255, 255, 255))
        print(np.unique(mask))
    json_filename_noex = Path(json_filename).stem
    mask_img_filename = target_dirname + '/' + json_filename_noex + "_mask.png"

    # save the mask 
    cv2.imwrite(mask_img_filename, mask)

def main():
    for dirname, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if fnmatch.fnmatch(filename,"*.json") == True:
                relative_filename = os.path.join(dirname,filename)
                relative_filename = relative_filename.replace("\\","/")
                print("generate mask img for " + relative_filename)
                gen_mask_img(relative_filename)
            else:
                pass

if __name__ == '__main__':
    main()
