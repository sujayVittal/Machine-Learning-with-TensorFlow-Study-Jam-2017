
# Initial Steps:
1. Analyze 'mnist_softmax.py' file 

  A. The MNIST database is hosted on http://yann.lecun.com/exdb/mnist/ [Download]
                            OR
  B. Run the following code to download and read in the data automatically:
  
      from tensorflow.examples.tutorials.mnist import input_data
      mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    
## PLEASE NOTE: 
  
    The MNIST data is split into three parts: 
    a. 55,000 data points of Training data - mnist.train
    b. 10,000 data points of Test data - mnist.test
    c. 5,000 data points of Validation data - mnist.validation

    Every MNIST data point has two parts: an image of a handwritten digit and a corresponding label. We'll call the images "x" and the labels "y". 
    Both the training set and test set contain images and their corresponding labels.
    For example the training images are mnist.train.images and the training labels are mnist.train.labels.
