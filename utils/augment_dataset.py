import os
import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import shutil
from PIL import Image


MAX_IMAGES = 3000 # 3000 images per class

if __name__ == '__main__':

    train_dir = 'train'
    output_dir = 'train_large'
    cwd = os.getcwd()

    if not os.path.isdir(os.path.join(cwd, output_dir)):
        os.makedirs(os.path.join(cwd, output_dir))

    new_dir = os.path.join(cwd, output_dir)


    augmentations = transforms.Compose([
      transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
      transforms.RandomRotation(degrees=(0, 180)),
      transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.75, 0.75)),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomVerticalFlip(p=0.5)
    ])


    for label in os.listdir(train_dir):

        if not os.path.isdir(os.path.join(new_dir, label)):
            os.makedirs(os.path.join(new_dir, label))

        src_dir = os.path.join(train_dir, label)
        dst_dir = os.path.join(new_dir, label)

        for file in os.listdir(src_dir):

            src = os.path.join(src_dir, file)
            dst = os.path.join(dst_dir, file)
            shutil.copyfile(src, dst)


    for label in os.listdir(new_dir):
        if not os.path.isdir(os.path.join(new_dir, label)):
            os.makedirs(os.path.join(new_dir, label))
            
        label_dir = os.path.join(new_dir, label)
        file_list = os.listdir(label_dir)
        images_required = MAX_IMAGES - len(file_list)
        start = 0
        end = len(file_list)

        while images_required > 0:
            start = start % end
            image_loc = file_list[start]

            image = Image.open(os.path.join(label_dir, image_loc)).convert("RGB")

            image = augmentations(image)

            image = np.array(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cv2.imwrite(filename=os.path.join(label_dir, f'augmented_{image_loc}_{images_required}.png'), img=image)
            images_required -= 1
            start += 1

    
    for label in os.listdir(new_dir):
        file_list = os.listdir(os.path.join(new_dir, label))
        assert len(file_list) == MAX_IMAGES
        print(f'{label} has {MAX_IMAGES} images')
