#import matplotlib.image as mpimg
import numpy as np
import cv2
import pandas
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import math
import argparse
import os
import utils
import model
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Train a car to drive itself')
parser.add_argument( '--data_dir', type=str, default='./data', 
    help='Path to Image directory and Driving log CSV file' )

parser.add_argument( '--steering_correction', type=float, default=0.25, 
    help='Steering Correction to applied to left and right camera images' )

parser.add_argument( '--epochs', type=int, default=5, 
    help='Number of Epochs to train the model+' )

parser.add_argument( '--batch_size', type=int, default=128, 
    help='Batch Size for training.' )

parser.add_argument( '--model_path', type=str, default='./model.h5', 
    help='Batch Size for training.' )

args = parser.parse_args()

data_dir = args.data_dir
steering_correction = args.steering_correction
epochs = args.epochs
batch_size = args.batch_size

# Create column names for the Driving log CSV data
col_names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

# Read the Driving log CSV data
csv_data = pandas.read_csv(os.path.join(data_dir,'driving_log.csv'), skiprows=[0], names=col_names)

# Convert all the data to list
center_images = csv_data.center.tolist()
left_images = csv_data.left.tolist()
right_images = csv_data.right.tolist()
steering_angles = csv_data.steering.tolist()

# Shuffle the data
center_images_randomised, steering_angles_randomised = shuffle(center_images, steering_angles)

# Split the data into training and validation set
center_images_train, center_images_test, steering_angles_train, steering_angles_test = train_test_split(center_images_randomised, steering_angles_randomised, test_size = 0.20, random_state = 10) 

print ("Number of training samples before augmentation: "+str(len(center_images_train)))
del(center_images_randomised)
del(steering_angles_randomised)


# Check and show the train, validation data steering feature
def show_steering(y_train, y_valid):
    '''take train and validation data label-steering and visualize a histogram.
    input: y_train : train set label,
           y_valid : validation set label,
    output: Histogram of labels'''
        
    max_degree = 25
    degree_per_steering =10
    n_classes = max_degree * degree_per_steering
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    plt.subplots_adjust(left=0, right=0.95, top=0.9, bottom=0.25)
    ax0, ax1= axes.flatten()

    ax0.hist(y_train, bins=n_classes, histtype='bar', color='blue', rwidth=0.6, label='train')
    ax0.set_title('Number of training')
    ax0.set_xlabel('Steering Angle')
    ax0.set_ylabel('Total Image')

    ax1.hist(y_valid, bins=n_classes, histtype='bar', color='red', rwidth=0.6, label='valid')
    ax1.set_title('Number of validation')
    ax1.set_xlabel('Steering Angle')
    ax1.set_ylabel('Total Image')

    fig.tight_layout()
    plt.show()
     
show_steering(steering_angles, steering_angles)


steer_straight_images = [] 
steer_left_images = [] 
steer_right_images = []
steer_straight_angles = []
steer_left_angles = [] 
steer_right_angles = []

ignore_zero_steering = True
for idx, steering_angle in enumerate(steering_angles_train):
    if steering_angle > 0.15:
        steer_right_images.append(center_images_train[idx])
        steer_right_angles.append(steering_angle)

    elif steering_angle < -0.15:
        steer_left_images.append(center_images_train[idx])
        steer_left_angles.append(steering_angle)

    elif ignore_zero_steering:
        continue

    else:
        steer_straight_images.append(center_images_train[idx])
        steer_straight_angles.append(steering_angle)


left_image_diff = len(steer_straight_images) - len(steer_left_images)
right_image_diff = len(steer_straight_images) - len(steer_right_images)

print ("The difference in Left steering images and Right steering images are: "+str(left_image_diff)+" and "+str(right_image_diff)+" respectively")
len_center_image  = len(center_images) 

print ("Left steering images before adding right camera images: "+str(len(steer_left_images)))
idx = 0
while (left_image_diff > 0 and idx < len_center_image):
    if steering_angles[idx] < -0.15: 
        steer_left_images.append(right_images[idx])
        steer_left_angles.append(steering_angles[idx] - args.steering_correction)
        left_image_diff -= 1
    idx += 1

print ("Left steering images after adding right camera images: "+str(len(steer_left_images)))

print ("Right steering images before adding left camera images: "+str(len(steer_right_images)))
idx = 0
while (right_image_diff > 0 and idx < len_center_image):
    if steering_angles[idx] > 0.15: 
        steer_right_images.append(left_images[idx])
        steer_right_angles.append(steering_angles[idx] + args.steering_correction)
        right_image_diff -= 1
    idx += 1
print ("Right steering images after adding left camera images: "+str(len(steer_right_images)))


print ("Loading center camera images.")
steer_straight_images, steer_straight_angles = utils.load_images(steer_straight_images, steer_straight_angles)
print ("Loading left camera images.")
steer_left_images, steer_left_angles = utils.load_images(steer_left_images, steer_left_angles,flip=True)
print ("Loading right camera images.")
steer_right_images, steer_right_angles = utils.load_images(steer_right_images, steer_right_angles, flip=True)
print ("Done")

center_images_test, steering_angles_test = utils.load_images(center_images_test, steering_angles_test)

shuffle(steer_straight_images,steer_straight_angles)

steer_straight_images = steer_straight_images[:len(steer_left_images)+len(steer_right_images)]
steer_straight_angles = steer_straight_angles[:len(steer_left_images)+len(steer_right_images)]

if len(steer_straight_images>0):
    center_images_train = np.concatenate([steer_straight_images, steer_left_images, steer_right_images],axis=0)
    steering_angles_train =  np.float32(steer_straight_angles + steer_left_angles + steer_right_angles)
else:
    center_images_train = np.concatenate([ steer_left_images, steer_right_images],axis=0)
    steering_angles_train =  np.float32(steer_left_angles + steer_right_angles)

shuffle(center_images_train,steering_angles_train)
print ("Total number of training samples: ",str(len(center_images_train)))
print ("Training samples shape: ",str(center_images_train.shape))
print ("Training samples labels shape: "+str(steering_angles_train.shape))

show_steering(steering_angles_train, steering_angles_test)


def data_generator(samples, samples_labels, batch_size=128):
    num_samples = len(samples)
    choice = [True, False]
    while True:
        shuffle(samples,samples_labels)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:(offset + batch_size)]
            batch_samples_labels = samples_labels[offset:(offset + batch_size)]
            
            X_train, y_train = np.array(batch_samples), np.array(batch_samples_labels)
            yield shuffle(X_train, y_train)


def train_generator(samples, samples_labels, batch_size=128):
    return data_generator(samples, samples_labels, batch_size=batch_size)

def valid_generator(samples, samples_labels, batch_size=128):
    return data_generator(samples, samples_labels, batch_size=batch_size)

model = model.model()

train_generator = train_generator(samples= center_images_train, samples_labels= steering_angles_train, batch_size= args.batch_size)
valid_generator = valid_generator(samples= center_images_test, samples_labels= steering_angles_test, batch_size= args.batch_size)

model.fit_generator(train_generator, samples_per_epoch = len(center_images_train), nb_epoch=args.epochs, validation_data = valid_generator, nb_val_samples = len(center_images_test))

print('Done Training')

print ("Saving the model")
model.save(args.model_path)
print ("Saved the model to: "+str(args.model_path))