import cv2
import numpy as np
import random

def bgr2rgb(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def flip_image(image):
    return cv2.flip(image, 1)

def crop_image(image):
    cropped = image[60:140, :]
    return cropped

def resize(image, shape=(200, 66)):
    return cv2.resize(image, shape, interpolation = cv2.INTER_AREA)

def crop_and_resize(image):
    cropped = crop_image(image)
    resized = resize(cropped)
    return resized

def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    rand = random.uniform(0.3,1.0)
    hsv[:,:,2] = rand*hsv[:,:,2]
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return image
    
def blur_image(image):
	return cv2.GaussianBlur(image, (3,3), 0)


def load_images(image_path,steer_angles,flip=False):
    images = []
    angles = []
    for idx in range(len(image_path)):
        image = cv2.imread(image_path[idx])
        rgb_image = bgr2rgb(image)
        resized = crop_and_resize(rgb_image)
        images.append(resized)
        angles.append(steer_angles[idx])
        if flip == True:
            images.append(flip_image(resized))
            angles.append(steer_angles[idx] * (-1.))

    return np.float32(images), (angles)