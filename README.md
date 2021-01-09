a collaborative project between github users "hendricksgin" and "jingweisim"

# doggo-bot
Hack&amp;Roll solution 2021. Dog breed classifier on telegram bot.

#Telegram Bot Dog Breed Image Classification with Tensorflow & Telegram

The image classification system is implemented with Tensorflow (TF) and trained on the Stanford Dogs Dataset.
The Neural Network (NN) comprises of Transfer Learning with a Convolutional NN model architecture. 
The model was trained with 12,000 training images which comprises of 120 classes (dog breeds), obtaining an accuracy of 92.015 on 3 epoch due to time contrains of the 24 hrs hackathon.


#Get dataset

To install the dataset used for the training of the model, head over to http://vision.stanford.edu/aditya86/ImageNetDogs/ and click on the "Images(757MB)" under "Download". 
The images will be downloaded as a tar file which can be extracted using a zip extractor software.
Specify the dataset path that you have downloaded in the model.py file.

#Specify Token

Copy and past your token in the config.py file in order for the telegram bot to configure.

#References:
Transfer Learning with Tensorflow Hub
https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub

How to Deploy a Telegram Bot using Heroku for FREE
https://towardsdatascience.com/how-to-deploy-a-telegram-bot-using-heroku-for-free-9436f89575d2

Learn to build your first bot in Telegram with Python
https://www.freecodecamp.org/news/learn-to-build-your-first-bot-in-telegram-with-python-4c99526765e4/

#Team Background
A team of two university sophomore students who are just starting out in programming, data science and machine learning.
