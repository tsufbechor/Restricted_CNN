
Introduction:
=

In this project I am training a CNN on the Cifar10 dataset. 
I aim to achieve 80%+ accuracy on the test set.
I am limiting the model to a maximum amount of 50,000 parameters. 
Challenges:
=
Most state of the art CNN NN have >>>50,000 parameters, so taking inspiration from these models would be difficult.
Adding more layers to our NN could help achieve 80% accuracy but each layer would add parameters to the model.
There is a definite tradeoff between the complexity of the model (#parameters) and the accuracy.
Method:
=
First of all the model used contains 45,420 parameters.
The model contains 7 convolutional layers, of which two are FCN layers- convolutional layers with 
kernel size=1. I used these layers as an alternative to a Fully-Connected layer at the end as I found 
the Fully-Connected layer contains a large amount of parameters and withdrawing this layer allows 
more Convolutional Layers in the model under the constraint on the number of parameters which 
led to better performance for the model.

The first layer:
= 
Input channels-3, Output channels=32,padding=1,kernel size=3.
Then, we normalized the output of this layer with BatchNorm2d, broke the linearity with the ReLu 
activation function and finally applied Max Pooling(2,2).
The second layer:
=
Input channels-32, Output channels=64 ,padding=1,kernel size=3.
Then, we normalized the output of this layer with BatchNorm2d, broke the linearity with the ReLu 
activation function and finally applied Max Pooling(2,2).
The third layer:
=
Input channels-64, Output channels=32, padding=1,kernel size=3.
Then, we normalized the output of this layer with BatchNorm2d, broke the linearity with the ReLu 
activation function and finally applied Max Pooling(2,2).
The fourth layer:
=
Input channels-32, Output channels=16,padding=1,kernel size=3.
Then, we normalized the output of this layer with BatchNorm2d, broke the linearity with the ReLu 
activation function and finally applied Max Pooling(2,2).
The fifth layer:
=
Input channels-16, Output channels=16,padding=1,kernel size=3.
Then, we normalized the output of this layer with BatchNorm2d, broke the linearity with the ReLu 
activation function and finally applied Max Pooling(2,2).
The sixth layer:
=
Input channels-16, Output channels=10,padding=1,kernel size=1.
Then, we normalized the output of this layer with BatchNorm2d, broke the linearity with the ReLu 
activation function and finally applied Max Pooling(2,2).
This layer is a FCN layer.
The seventh layer:
=
Input channels-10, Output channels=10,padding=1,kernel size=1.
Then, we applied Max Pooling(2,2). This layer is a FCN layer.
I found the model did not overfit and after trying to apply Drop-Out in a few different variation, I 
found that the performance of the model on both the train and validation sets deteriorated.
Thus, I decided against using dropout in the model. After trying different learning rates, I found the 
model that yielded the best performance was with learning rate 0.001. Batch size-128.
