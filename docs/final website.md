# Facial Expression Recognition

Group members: Zheng Wang, Tianchen Shen, Jason Li

## Description

We implemented facial expression recognition project to use a facial expression dataset from Kaggle, and train a convolution neural network to classify facial expressions from input images. We also implemented a script that uses webcam to capture images, and feed the image into the trained neural network to do real time recognition. 

## Previous work
We used a couple of different inspirations for our approach.

We borrowed some ideas from an existing project that participate in kaggle’s facial expression recognition competition (https://github.com/jaydeepthik/kaggle-facial-expression-recognition).

We also implemented some techniques from the tutorials on the class website to improve our model.

We used the "cropping face" function from an online tutorial (https://pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/) to automatically detect and align faces in images.

## Our approach
We applied machine learning in this classification task as well as try different structures of neural networks to compare the results, including simple neural network with three layers, convolution neural network, and convolution neural network with batch normalization. We applied various techniques, including simulated annealing, weight decay, data augmentation, and combining similar image groups. We also tried varying the model parameters such as learning rate, weight decay, functions, etc. for optimizing results.

## Setup
In this project, we successfully applied neural networks to recognize facial expressions. The categories include 7 expressions, anger, disgust, fear, happy, neutral, sad, and surprise. By running the code provided in the jupyter notebook can obtain the trained networks. There are several different networks, and the best one could be saved for real-time recognition use.  
  
The in the app script, we can apply the best trained result as neural network parameters. By running the script, the camera will capture images all the time, automatically crop out the face from the captured image, and print the predicted results into the console. 

## Datasets
The dataset we use is found on Kaggle website, https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset. This is an labeled dataset that divides the data into train and test portions, and images are labeled by categories. 

## Results
network									training accuracy		testing  accuracy
---------------------------------------------------------------------------------
simple network							0.248569				0.258279
cov network								0.439645				0.403765
cov with SA								0.393047				0.386782
BNcov									0.908539				0.467167
BNcov with SA							0.768953				0.504246
BNcov with SA and WD					0.709968				0.508491
BNcov with SA, WD, and DA				0.563790				0.492499
Combine (angry,disgust,fear,surprise)  	0.716318				0.582366
Combine (disgust,fear,surprise) 		0.544985				0.493348

## Discussion
### What problems did you encounter?
1. Over fitting: We found that for particular models, the training accuracy can be much higher than the testing accuracy. So it is very difficult to generalize the model for images not in the dataset. 

2. Hard to detect similar expressions: We found that for some facial expressions, the model have difficulties distinguishing between the two. For example, the surprise and fear expression. The images from the dataset for these two expressions both have big open mouth, wide open eyes, and raised eyebrow. 

### Are there next steps you would take if you kept working on the project?
1. From other's work on Kaggle, github, the test accuracy are generally around 60%, and our accuracy is around 50%, so there is still space for improvement. We might try different neural network structure. 
2. Transfer learning could also be a good approach for this project, since those networks can be fine tuned. 
3. Clean up dataset. There are some images only captured side face, and a few images are even not human faces. We didn't remove them in this project, since there are only very small amount, so we assume they should not have a huge effect on the model. However, removing them could be another improvement. 

### How does your approach differ from others? Was that beneficial?
The approach are generally similar to others' work, since we all focus on constructing convolutional neural networks, and it is believed to be an very effective model for this project. Our approach implement different model to solve the problem mention above that may be different from other approaches. We would say this can be beneficial since for this particular model, it is hard to reach a high testing accuracy, and we think it depends on the the structure of layers of neural network. 
