# Training

#### 1. To implement cross-entropy we need to first add a new placeholder to input the correct answers:
     
      y_ = tf.placeholder(tf.float32, [None, 10])
     
#### 2. Then we can implement the cross-entropy function, −∑y′log⁡(y):

      cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
         
######   First, tf.log computes the logarithm of each element of y. Next, we multiply each element of y_ with the corresponding element of tf.log(y). Then tf.reduce_sum adds the elements in the second dimension of y, due to the reduction_indices=[1] parameter. Finally, tf.reduce_mean computes the mean over all the examples in the batch.
   
######   Note that in the source code, we don't use this formulation, because it is numerically unstable. Instead, we apply tf.nn.softmax_cross_entropy_with_logits on the unnormalized logits (e.g., we call softmax_cross_entropy_with_logits on tf.matmul(x, W) + b), because this more numerically stable function internally computes the softmax activation. In your code, consider using tf.nn.softmax_cross_entropy_with_logits instead.
   
######   Now that we know what we want our model to do, it's very easy to have TensorFlow train it to do so. Because TensorFlow knows the entire graph of your computations, it can automatically use the backpropagation algorithm to efficiently determine how your variables affect the loss you ask it to minimize. Then it can apply your choice of optimization algorithm to modify the variables and reduce the loss.
   
      train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
      
######   In this case, we ask TensorFlow to minimize cross_entropy using the gradient descent algorithm with a learning rate of 0.5. Gradient descent is a simple procedure, where TensorFlow simply shifts each variable a little bit in the direction that reduces the cost. But TensorFlow also provides many other optimization algorithms: using one is as simple as tweaking one line.

######   What TensorFlow actually does here, behind the scenes, is to add new operations to your graph which implement backpropagation and gradient descent. Then it gives you back a single operation which, when run, does a step of gradient descent training, slightly tweaking your variables to reduce the loss.


#### 3. We can now launch the model in an InteractiveSession:

      sess = tf.InteractiveSession()
      
#### 4. We first have to create an operation to initialize the variables we created:

      tf.global_variables_initializer().run()
      
#### 5. Let's train -- we'll run the training step 1000 times!

      for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        
######   Each step of the loop, we get a "batch" of one hundred random data points from our training set. We run train_step feeding in the batches data to replace the placeholders.

######   Using small batches of random data is called stochastic training -- in this case, stochastic gradient descent. Ideally, we'd like to use all our data for every step of training because that would give us a better sense of what we should be doing, but that's expensive. So, instead, we use a different subset every time. Doing this is cheap and has much of the same benefit.
 
   
# Evaluating our Model

## How well does our model do? 

#### 1. First let's figure out where we predicted the correct label. tf.argmax is an extremely useful function which gives you the index of the highest entry in a tensor along some axis. For example, tf.argmax(y,1) is the label our model thinks is most likely for each input, while tf.argmax(y_,1) is the correct label. We can use tf.equal to check if our prediction matches the truth:

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        
#### 2. That gives us a list of booleans. To determine what fraction are correct, we cast to floating point numbers and then take the mean. For example, [True, False, True, True] would become [1,0,1,1] which would become 0.75.

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
#### 3. Finally, we ask for our accuracy on our test data.

        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        
        
### This should be about **_92%_**.
