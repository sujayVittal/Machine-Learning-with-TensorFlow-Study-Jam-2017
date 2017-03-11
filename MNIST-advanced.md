# MNIST Advanced - Building a multilayer convolutional Network

##### 1. Load MNIST Data:
    
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    

##### 2. Start TensorFlow InteractiveSession:

    import tensorflow as tf
    sess = tf.InteractiveSession()
    

##### 3. We start building the computation graph by creating nodes for the input images and target output classes:

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    
   Here _*x*_ and _*y*_ aren't specific values. Rather, they are each a placeholder *_-- a_* value that we'll input when we ask TensorFlow to run a computation.
    

##### 4. We now define the weights _*W*_ and biases _*b*_ for our model. We could imagine treating these like additional inputs, but TensorFlow has an even better way to handle them: Variable. A Variable is a value that lives in TensorFlow's computation graph. It can be used and even modified by the computation. In machine learning applications, one generally has the model parameters be Variables.

    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    

##### 5. We can now implement our regression model. It only takes one line! We multiply the vectorized input images _*x*_ by the weight matrix _*W*_, add the bias _*b*_.

    y = tf.matmul(x,W) + b
    
   We can specify a loss function just as easily. Loss indicates how bad the model's prediction was on a single example; we try to minimize that while training across all the examples. Here, our loss function is the cross-entropy between the target and the softmax activation function applied to the model's prediction. As in the beginners tutorial, we use the stable formulation:
   
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    
   Note that _*tf.nn.softmax_cross_entropy_with_logits*_ internally applies the softmax on the model's unnormalized model prediction and sums across all classes, and tf.reduce_mean takes the average over these sums.
   

##### 6. Now that we have defined our model and training loss function, it is straightforward to train using TensorFlow. Because TensorFlow knows the entire computation graph, it can use automatic differentiation to find the gradients of the loss with respect to each of the variables. TensorFlow has a variety of built-in optimization algorithms. For this example, we will use steepest gradient descent, with a step length of 0.5, to descend the cross entropy.

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    
   What TensorFlow actually did in that single line was to add new operations to the computation graph. These operations included ones to compute gradients, compute parameter update steps, and apply update steps to the parameters.
   
So we first run the train step:

    tf.initialize_all_variables().run()
   
   The returned operation _*train_step*_, when run, will apply the gradient descent updates to the parameters. Training the model can therefore be accomplished by repeatedly running _*train_step*_.

    for i in range(1000):
      batch = mnist.train.next_batch(100)
      train_step.run(feed_dict={x: batch[0], y_: batch[1]})
      
   We load 100 training examples in each training iteration. We then run the train_step operation, using _*feed_dict*_ to replace the placeholder tensors _*x*_ and _*y_ *_ with the training examples. Note that you can replace any tensor in your computation graph using feed_dict -- _it's not restricted to just placeholders_.
   
   
##### 7. How well did our model do? First we'll figure out where we predicted the correct label. _*tf.argmax*_ is an extremely useful function which gives you the index of the highest entry in a tensor along some axis. For example, _*tf.argmax(y,1)*_ is the label our model thinks is most likely for each input, while _*tf.argmax(y_,1)*_ is the true label. We can use _*tf.equal*_ to check if our prediction matches the truth.

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    
  That gives us a list of booleans. To determine what fraction are correct, we cast to floating point numbers and then take the mean. For example, _*[True, False, True, True]*_ would become _*[1,0,1,1]*_ which would become _*0.75*_.
  
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
  Finally, we can evaluate our accuracy on the test data. This should be about _*92%*_ correct.
  
    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    
    
##### 8. *Building a Multilayer Convolutional Network*

##### Getting a *92%* accuracy on MNIST is bad. Now, let's fix that and get higher accuracy rate.

###### A. Weight Initialization:
   
   To create this model, we're going to need to create a lot of weights and biases. One should generally initialize weights with a small amount of noise for symmetry breaking, and to prevent 0 gradients. Since we're using *ReLU* neurons, it is also good practice to initialize them with a slightly positive initial bias to avoid "dead neurons". Instead of doing this repeatedly while we build the model, let's create two handy functions to do it for us.
    
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)
      
###### B. Convolution and Pooling:

   TensorFlow also gives us a lot of flexibility in convolution and pooling operations. How do we handle the boundaries? What is our stride size? In this example, we're always going to choose the vanilla version. Our convolutions uses a stride of one and are zero padded so that the output is the same size as the input. Our pooling is plain old max pooling over 2x2 blocks. To keep our code cleaner, let's also abstract those operations into functions.
   
    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
      
###### C. First Convolution Layer:

   We can now implement our first layer. It will consist of convolution, followed by max pooling. The convolution will compute 32 features for each 5x5 patch. Its weight tensor will have a shape of [5, 5, 1, 32]. The first two dimensions are the patch size, the next is the number of input channels, and the last is the number of output channels. We will also have a bias vector with a component for each output channel.
   
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    
   To apply the layer, we first reshape *_x_* to a 4d tensor, with the second and third dimensions corresponding to image width and height, and the final dimension corresponding to the number of color channels.
   
    x_image = tf.reshape(x, [-1,28,28,1])
    
   We then convolve _*x_image*_ with the weight tensor, add the bias, apply the ReLU function, and finally max pool. The *_ max_pool_2x2 _* method will reduce the image size to _*14x14*_.
   
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
###### D. Second Convolution Layer:

   In order to build a deep network, we stack several layers of this type. The second layer will have 64 features for each 5x5 patch.
   
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
###### E. Densely Connected Layer:

   Now that the image size has been reduced to 7x7, we add a fully-connected layer with 1024 neurons to allow processing on the entire image. We reshape the tensor from the pooling layer into a batch of vectors, multiply by a weight matrix, add a bias, and apply a ReLU.
   
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
###### F. Dropout:

   To reduce overfitting, we will apply *dropout* before the readout layer. We create a *_placeholder_* for the probability that a neuron's output is kept during dropout. This allows us to turn dropout on during training, and turn it off during testing. TensorFlow's *_tf.nn.dropout op automatically handles scaling neuron outputs in addition to masking them, so dropout just works without any additional scaling.
   
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
###### G. Readout:

   Finally, we add a layer, just like for the one layer softmax regression above.
   
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    
##### 9. Train and Evaluate the Model:

How well does this model do? To train and evaluate it we will use code that is nearly identical to that for the simple one layer SoftMax network above.

The differences are that:
  We will replace the steepest gradient descent optimizer with the more sophisticated ADAM optimizer.
  We will include the additional parameter keep_prob in feed_dict to control the dropout rate.
  We will add logging to every 100th iteration in the training process.
  
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
      batch = mnist.train.next_batch(50)
      if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
        
##### The final test set accuracy after running this code should be approximately *_99.2%_*.
