
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

Model:
=
First of all the model used contains 45,420 parameters.
The model contains 7 convolutional layers, of which two are FCN layers- convolutional layers with 
kernel size=1. I used these layers as an alternative to a Fully-Connected layer at the end as I found 
the Fully-Connected layer contains a large amount of parameters and withdrawing this layer allows 
more Convolutional Layers in the model under the constraint on the number of parameters which 
led to better performance for the model. The other Conv layers were with a kernel size=3

Image Augmentation:
=
After plateauing at around 75% with different modifications to the Model architecture, I tried to increase the dataset the model is trained on.

I took the original training dataset and executed random rotations,flips,crops,etc. in order to generate more training examples for the model to learn.

Increasing the number of examples trained on, had a tangible effect on the performance and 
generalization of the model.

Results:
=
![image](https://user-images.githubusercontent.com/81694762/222185451-79b7719a-fb0b-4725-a7ab-2fe8816d39d4.png)
![image](https://user-images.githubusercontent.com/81694762/222185545-4756fee3-cf94-4be0-876b-ac5bbd610139.png)






