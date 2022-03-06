import cv2
import numpy as np
import os


class PreProcess():
    
    def create_mask_for_plant(self, image):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sensitivity = 35

        # Get pixels from image in the following (Hue, Saturation, Lightness) range
        lower_hsv = np.array([60 - sensitivity, 100, 50])
        upper_hsv = np.array([60 + sensitivity, 255, 255])

        mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def segment_plant(self, image):
        mask = self.create_mask_for_plant(image)
        output = cv2.bitwise_and(image, image, mask = mask)
        return output

    def sharpen_image(self, image):
        image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
        image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
        return image_sharp
    
    def __call__(self, image):
        image = self.segment_plant(image)
        return self.sharpen_image(image)

if __name__ == '__main__':
    train_dir = 'train_large'
    destination_dir = 'train_large_seg'
    test_dir = 'test'

    preprocessor = PreProcess()


    for Class in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(destination_dir, Class)):
            os.makedirs(os.path.join(destination_dir, Class))
        class_dir = os.path.join(train_dir, Class)
        for location in os.listdir(class_dir):
            image = cv2.imread(os.path.join(class_dir, location), cv2.IMREAD_COLOR)
            image = preprocessor(image=image)
            destination = destination_dir + '/' + Class + '/' + location
            cv2.imwrite(destination, image)

    for location in os.listdir(test_dir+'/'):
        image = cv2.imread(test_dir+'/'+location, cv2.IMREAD_COLOR)
        image = preprocessor(image=image)
        destination = 'test_seg/' + location
        cv2.imwrite(destination, image)

