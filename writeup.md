# **Traffic Sign Classifier** 
 

---

<img src="https://github.com/BVG85/Project-2-Traffic-Sign-Classifier/blob/master/sign.jpg">


### Writeup / README

This file is the writeup for the Traffic Sign Classifier project as part of Udacity's Self-driving Car Engineer Nanodegree.
The implementation of the various portions is discussed to address the rubic points.

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

Project code can be found [here](https://github.com/BVG85/Project-2-Traffic-Sign-Classifier/blob/master/traffic_sign_classifier.ipynb)

### Data Set Summary & Exploration

The Numpy library was used to calculate the following summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### Visualization of the dataset.

Here is an exploratory visualization of the data set. It shows a random traffic sign, along with the index number

![alt text](https://github.com/BVG85/Project-2-Traffic-Sign-Classifier/blob/master/vis.png)

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
- validation set accuracy of 96.7 %
- test set accuracy of 95.3 %

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
 
<img src="https://github.com/BVG85/Project-2-Traffic-Sign-Classifier/blob/master/new/1.jpg" width="120" height="120"> <img src="https://github.com/BVG85/Project-2-Traffic-Sign-Classifier/blob/master/new/2.jpg" width="120" height="120"> <img src="https://github.com/BVG85/Project-2-Traffic-Sign-Classifier/blob/master/new/3.jpg" width="120" height="120"> <img src="https://github.com/BVG85/Project-2-Traffic-Sign-Classifier/blob/master/new/4.jpg" width="120" height="120">
<img src="https://github.com/BVG85/Project-2-Traffic-Sign-Classifier/blob/master/new/5.jpg" width="120" height="120"> <img src="https://github.com/BVG85/Project-2-Traffic-Sign-Classifier/blob/master/new/6.jpg" width="120" height="120"> <img src="https://github.com/BVG85/Project-2-Traffic-Sign-Classifier/blob/master/new/7.jpg" width="120" height="120"> <img src="https://github.com/BVG85/Project-2-Traffic-Sign-Classifier/blob/master/new/8.jpg" width="120" height="120">

The fourth image may be difficult to classify and seventh images maybe difficult to classify as the sign itself is small within the image. Also, five and seven look similar, so the model may have some difficulty with these images.

#### Predictions on new traffic signs 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection      		| End of no passing  									| 
| Go straight or right    			| Go straight or right										|
| No vehicles				| No vehicles											|
| Speed limit (30km/h)      		| Speed limit (30km/h)					 				|
| Slippery road		| Slippery Road      							|
| Stop		| Stop     							|
| Speed limit (50km/h)			| Speed limit (30km/h)     							|
| Double curve			| Double curve     							|

The model was able to correctly guess 6 of the 8 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 75%. On previous iterations of the model 87.5% and 100% accuraries were obtained.

#### The top 5 softmax probabilities for each image along with the sign type of each probability. 

For the first image, the model has a very high probability for "end of no passing" sign, however the image contains a "right-of-way sign at the next intersection" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99646187e-01        			| End of no passing    									| 
| 3.53865616e-04     				| End of all speed and passing limits										|
| 5.23706035e-12					| Priority road									|
| 3.34431723e-12      			| Roundabout mandatory				 				|
| 2.79893274e-12			    | Dangerous curve to the right      							|

For the second image, the model predicted the sign correctly:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00        			| Go straight or right   									| 
| 3.90913001e-24   				| Keep right									|
| 1.19299459e-25					| Turn left ahead								|
| 9.23018412e-26     			| No vehicles		 				|
| 6.99420410e-26		    | End of all speed and passing limits     							|

For the third image, the model predicted the sign correctly:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00        			| No vehicles   									| 
| 1.17959672e-15   				| Yield									|
| 1.96194767e-17					| Speed limit (30km/h)								|
| 6.13155177e-19     			| No passing	 				|
| 2.71654385e-20		    | Speed limit (30km/h)    							|

For the fourth image, the model predicted the sign correctly. All other predictions were speed limits as well:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00        			| Speed limit (30km/h)  									| 
| 2.14057949e-25  				| Speed limit (50km/h)							|
| 3.21331784e-32			| Speed limit (80km/h)								|
| 0.00000000e+00     			| Speed limit (20km/h)				|
| 0.00000000e+00		    | Speed limit (60km/h)   							|

For the fifth image, the model predicted the sign correctly:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00        			| Slippery road  									| 
| 3.15522746e-20   				| Road work									|
| 1.71059254e-20					| Beware of ice/snow							|
| 1.56477443e-24     			| Dangerous curve to the left	 				|
| 8.28894826e-27		    | Double curve    							|

For the sixth image, the model predicted the sign correctly:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00        			| Stop  									| 
| 1.71826014e-13   				| Speed limit (20km/h)								|
| 8.57121448e-14					| Speed limit (30km/h)					|
| 2.64993903e-23     			| No passing 				|
| 9.05511000e-26		    | No entry   							|

For the seventh image, the model predicted the sign incorrectly. This could be to the size of the sign within the image, as well as the similarity to the "Speed limit (30km/h)" sign:


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99929667e-01        			| Speed limit (30km/h)  									| 
| 7.03617479e-05  				| Speed limit (50km/h)							|
| 2.18761077e-12			| Keep right								|
| 4.13960451e-13     			| Roundabout mandatory				|
| 6.90928613e-17		    | Speed limit (80km/h)  							|

For the eighth image, the model predicted the sign correctly:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00        			| Double curve 									| 
| 5.14158247e-20  				| Beware of ice/snow								|
| 3.74433725e-21				| Wild animals crossing				|
| 1.55301193e-24     			| Bicycles crossing				|
| 2.58175611e-25		    | Go straight or left							|

### Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

The network feature maps react with high activation to the sign's outlines as well as to the contrast in the sign's painted symbol against the sign'c background:


![alt text](https://github.com/BVG85/Project-2-Traffic-Sign-Classifier/blob/master/fmap.png)

