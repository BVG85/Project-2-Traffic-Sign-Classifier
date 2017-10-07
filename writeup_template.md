#**Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

This file is the writeup for the Traffic Sign Classifier project as part of Udacity's Self-driving Car Engineer Nanodegree.
The implementation of the various portions is discussed to address the rubic points.

Project code can be found [here](https://github.com/BVG85/Project-2-Traffic-Sign-Classifier/blob/master/traffic_sign_classifier.ipynb)

### Data Set Summary & Exploration

The Numpy library was used to calculate the following summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It shows a random traffic sign, along with the index number

![alt text][image1]

### Design and Test a Model Architecture

The data set was normalized and shuffled during preprocessing.
 
Normalization is done to get all the data features on the same scale.
Shuffling is done on the training set as the order of the data may have an affect on how well the network trains.


#### Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| ReLU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 1x1     	| 1x1 stride, same padding, outputs 14x14x32 	|
| RELU					|												|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64      									|
| ReLU					|	
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				|
| Fully connected Layer 1		|  output 120  									|
| ReLU				|         									|
|	Dropout			|												|
|	Fully connected Layer 2					|		output 84										|
| ReLU				|         									|
|	Dropout			|												|
|	Fully connected Layer Output					|	output = number of classes									|

#### Training the model & Hyperparameters 

To train the model, the following hyperparameters were used:
 - 50 epochs
 - a batch size of 64
 - keep prob of 0.9 for dropout
 - a learning rate of 0.0005
 
For the optimizer AdamOptimizer was used.


#### Final results & approach to finding the solution Approach taken

The final model results were:
- validation set accuracy of 94.9 %
- test set accuracy of 94.1 %

The approach used was to start from the LeNet-5 architecture and adjust the model from there.
The LeNet-5 model was chosen, as the assignment indicated that an accuracy of about 89% was possible on the validation set.
For the German Traffic Sign dataset, this model was not deep or wide enough so the architecture was adjusted.

The following adjustments were made:

- The size of the convolutional layers were increased
- A 1x1 convolution was added between the convolutional layers
- Dropout was added to the two fully connected layers

- Then the hyper parameters were tuned, mainly focussing on learning rate and keep_prob for dropout.
- Later the batch size was decreased with good results.
 
### Test a Model on New Images


Eight German traffic signs were downloaded from the web:
 
<img src="https://github.com/BVG85/Project-2-Traffic-Sign-Classifier/blob/master/new/1.jpg" width="120" height="120"><img src="https://github.com/BVG85/Project-2-Traffic-Sign-Classifier/blob/master/new/2.jpg" width="120" height="120"><img src="https://github.com/BVG85/Project-2-Traffic-Sign-Classifier/blob/master/new/3.jpg" width="120" height="120"><img src="https://github.com/BVG85/Project-2-Traffic-Sign-Classifier/blob/master/new/4.jpg" width="120" height="120"><img src="https://github.com/BVG85/Project-2-Traffic-Sign-Classifier/blob/master/new/5.jpg" width="120" height="120"><img src="https://github.com/BVG85/Project-2-Traffic-Sign-Classifier/blob/master/new/6.jpg" width="120" height="120"><img src="https://github.com/BVG85/Project-2-Traffic-Sign-Classifier/blob/master/new/7.jpg" width="120" height="120"><img src="https://github.com/BVG85/Project-2-Traffic-Sign-Classifier/blob/master/new/8.jpg" width="120" height="120">

The fourth image may be difficult to classify and seventh images maybe difficult to classify as the sign itself is small within the image

#### Predictions on new traffic signs 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection      		| Right-of-way at the next intersection   									| 
| Go straight or right    			| Go straight or right										|
| No vehicles				| No vehicles											|
| Speed limit (30km/h)      		| Speed limit (30km/h)					 				|
| Slippery road		| Slippery Road      							|
| Stop		| Stop     							|
| Speed limit (50km/h)			| Speed limit (50km/h)     							|
| Double curve			| Double curve     							|

The model was able to correctly guess 8 of the 8 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.1 %

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


