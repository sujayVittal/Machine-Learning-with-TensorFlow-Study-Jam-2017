# _tf.contrib.learn_ hands-on

### The following code walk through is with respect to **_neural-network-classifer-iris.py_**. Please make sure you have the file. You can find the resouce file at:
  
    https://github.com/sujayVittal/Machine-Learning-with-TensorFlow-Study-Jam-2017/blob/master/neural-network-classifer-iris.py
  
### Source of data set
  1. A training set of 120 examples - http://download.tensorflow.org/data/iris_training.csv
  2. A test set of 30 samples - http://download.tensorflow.org/data/iris_test.csv
  
  Place these files in the **_same_** directory as your Python code!
  
#### 1. To get started, first import TensorFlow and NumPy:
    
    from __future__ import absolute_import
    from __future__ import division
    from __future__ import print_function

    import tensorflow as tf
    import numpy as np
    
### Load the Iris CSV data to TensorFlow:

#### 2. Next, load the training and test sets into Datasets using the load_csv_with_header() method in learn.datasets.base. The load_csv_with_header() method takes three required arguments:
 
  a. **filename**, which takes the filepath to the CSV file
  
  b. **target_dtype**, which takes the numpy datatype of the dataset's target value.

  c. **features_dtype**, which takes the numpy datatype of the dataset's feature values.
  
#### Here, the target (the value you're training the model to predict) is flower species, which is an integer from 0–2, so the appropriate numpy datatype is np.int:

    
    IRIS_TRAINING = "iris_training.csv"
    IRIS_TEST = "iris_test.csv"

    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TRAINING,
      target_dtype=np.int,
      features_dtype=np.float32)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)
      
      
#### Datasets in tf.contrib.learn are named tuples; you can access feature data and target values via the data and target fields. Here, training_set.data and training_set.target contain the feature data and target values for the training set, respectively, and test_set.data and test_set.target contain feature data and target values for the test set.  


### Construct a Deep Neural Network Classifier:

#### 3. f.contrib.learn offers a variety of predefined models, called Estimators, which you can use "out of the box" to run training and evaluation operations on your data. Here, you'll configure a Deep Neural Network Classifier model to fit the Iris data. Using tf.contrib.learn, you can instantiate your tf.contrib.learn.DNNClassifier with just a couple lines of code:

    
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]


    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[10, 20, 10],
                                                n_classes=3,
                                                model_dir="/tmp/iris_model")
                                                

#### 4. The code above first defines the model's feature columns, which specify the data type for the features in the data set. All the feature data is continuous, so tf.contrib.layers.real_valued_column is the appropriate function to use to construct the feature columns. There are four features in the data set (sepal width, sepal height, petal width, and petal height), so accordingly dimension must be set to 4 to hold all the data.

#### Then, the code creates a DNNClassifier model using the following arguments:

    a. feature_columns=feature_columns. The set of feature columns defined above.
    b. hidden_units=[10, 20, 10]. Three hidden layers, containing 10, 20, and 10 neurons, respectively.
    c. n_classes=3. Three target classes, representing the three Iris species.
    d. model_dir=/tmp/iris_model. The directory in which TensorFlow will save checkpoint data during model training. 
    

### Fit the DNNClassifier to the Iris Training Data

#### 5. Now that you've configured your DNN classifier model, you can fit it to the Iris training data using the fit method. Pass as arguments your feature data (training_set.data), target values (training_set.target), and the number of steps to train (here, 2000):
    
    classifier.fit(x=training_set.data, y=training_set.target, steps=2000)
    
#### 6. The state of the model is preserved in the classifier, which means you can train iteratively if you like. For example, the above is equivalent to the following:

    classifier.fit(x=training_set.data, y=training_set.target, steps=1000)
    classifier.fit(x=training_set.data, y=training_set.target, steps=1000)
    
#### However, if you're looking to track the model while it trains, you'll likely want to instead use a TensorFlow monitor to perform logging operations.


### Evaluate Model Accuracy

#### 7. You've fit your DNNClassifier model on the Iris training data; now, you can check its accuracy on the Iris test data using the evaluate method. Like fit, evaluate takes feature data and target values as arguments, and returns a dict with the evaluation results. The following code passes the Iris test data—test_set.data and test_set.target—to evaluate and prints the accuracy from the results:

    accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]
    print('Accuracy: {0:f}'.format(accuracy_score))
    
#### Run the script to check the accuracy results:
    
    Accuracy: 0.966667
    
    
  
