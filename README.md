# Image Classification using Convolution Neural Networks

## Dhyey Italiya (2021A7PS1463P)

## Abir Abhyankar (2021A7PS0523P)

This deep learning project aims to classify audio signals using Convolutional Neural Networks (CNNs), employing two prominent architectures: Residual Networks (ResNets) and GoogleNet.

## Getting Started

To get started with this project, follow the instructions below.

### Prerequisites

Please ensure you have Python installed on your system. We are using Python 3.9.7 for the project.

Run the **'Import_Libs.py'** File to install all the required dependencies.

We are using the libraries -

1. numpy
2. pandas
3. matplotlib
4. torch
5. torchvision
6. torch.optim
7. sklearn
8. seaborn
9. tqdm
10. librosa
11. pydub
12. soundfile
13. pillow
14. ffmpeg

Once you have installed the necessary libraries, you can execute the training and testing code.

We have 5 executables which you can run -

1. **image_converter.py** - This file takes in audio files dataset, and make the appropriate spectrogram dataset, on which we have trained both models. Please note that if you want to execute this file to view the spectrogram generation process, paste the audio dataset which was given, into this folder, and then you can run this file.
2. **DL_GoogleNet.py** - This file is used for training a GoogleNet Model for Audio Classication, you can directly run it to view the training process.
3. **DL_ResNet.py** - This file is used for training a ResNet Model for Audio Classication, you can directly run it to view the training process.
4. **testing_code_GoogleNet.py** - This file can be used for testing our GoogleNet Model and classify unseen audio files. Just instantiate **TEST_DATA_DIRECTORY_ABSOLUTE_PATH** and **OUTPUT_CSV** variables in this file, and you will get the output in the desired csv file.
5. **testing_code_ResNet.py** - This file can be used for testing our ResNet Model and classify unseen audio files. Just instantiate **TEST_DATA_DIRECTORY_ABSOLUTE_PATH** and **OUTPUT_CSV** variables in this file, and you will get the output in the desired csv file.

We have also included Loss, Validation Accuracy Curves along with Confusion Matrices for both of our models in the .zip file, and the final weights for this model, along with the provided dataset, and augmented dataset.

There are **2** files **ResNet_Module.py** and **GoogleNet_Module.py** which include the necessary classes for **ResNet** and **GoogleNet** respectively.

## THANK YOU!
