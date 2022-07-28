This folder contains some of the (examples of) files 
used during the machine learning part of the final master
project. In the data folder, scripts that can be used for
converting the saved data to usable training and test 
sets can be found. Below a list of the scripts in this 
folder can be found.

3D.ipynb 		Script to train and test the developed
				convolutional neural networks. 	
FCN.ipynb		Script to train and test the developed
				fully connected neural networks.
CoordConv.py	Script containing convolutional neural 
				network building blocks, containing different 
				implementations of the coordinate convolutional 
				layer using both PyTorch and Keras.
p3DNets.py		Script containing convolutional neural network
				building blocks. Example architectures given
				are ResNet-like and VGG-like ones. VGG-like nets
				are not recommended, due to the spatial information
				destroying pooling layers. The ResNet-like network
				without pooling is the best performing one, 
				currently.


It is recommended to move towards transformer architectures,
as the given networks all come with certain drawbacks.			
