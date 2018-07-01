# Autonomous Driving using Deep Learning and Behavioural Cloning
This repository is to support the code for [this](https://medium.com/@akarshzingade/autonomous-driving-using-deep-learning-and-behavioural-cloning-97983a57fe10) blog. It contains the code to train a Deep Neural Network to drive a car! 

## Introduction
There are 2 phases for this:
1) Training
2) Testing

## Training
The train.py file provides the code to train the Neural Network. 
```
usage: train.py [-h] [--data_dir DATA_DIR]
                [--steering_correction STEERING_CORRECTION] [--epochs EPOCHS]
                [--batch_size BATCH_SIZE] [--model_path MODEL_PATH]
                
  --data_dir DATA_DIR   Path to Image directory and Driving log CSV file
  --steering_correction STEERING_CORRECTION
                        Steering Correction to applied to left and right
                        camera images
  --epochs EPOCHS       Number of Epochs to train the model+
  --batch_size BATCH_SIZE
                        Batch Size for training.
  --model_path MODEL_PATH
                        Path to save the trained model.
```

## Testing 
The drive.py file provides the code to test the trained Neural Network.

```
usage: drive.py [-h] model [image_folder]

  model         Path to model h5 file. Model should be on the same path.
  image_folder  Path to image folder. This is where the images from the run
                will be saved.
```

## Credits
The drive.py is based on Udacity CarND Behavioral Cloning Project which can be found [here](https://github.com/udacity/CarND-Behavioral-Cloning-P3)
