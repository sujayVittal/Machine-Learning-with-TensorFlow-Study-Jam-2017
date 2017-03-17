# TensorBoard: Hands-On
#### The computations you'll use TensorFlow for - like training a massive deep neural network - can be complex and confusing. TensorBoard is a suite of Visualization tools that makes it easier to understand, debug, and optimize TensorFlow programs.

#### You can use TensorBoard to visualize your TensorFlow graph, plot quantitative metrics about the execution of your graph, and show additional data like images that pass through it. Configure TensorBoard and see how it looks :)

## Serializing the data:

##### TensorBoard operates by reading TensorFlow events files, which contain summary data that you can generate when running TensorFlow. Here's the general lifecycle for summary data within TensorBoard:
  
  ###### First, create the TensorFlow graph that you'd like to collect summary data from, and decide which nodes you would like to annotate with summary operations [Link: https://www.tensorflow.org/versions/master/api_docs/python/summary/].
  
  ###### For example, suppose you are training a convolutional neural network for recognizing MNIST digits. You'd like to record how the learning rate varies over time, and how the objective function is changing. Collect these by attaching scalar_summary ops to the nodes that output the learning rate and loss respectively. Then, give each scalar_summary a meaningful tag, like 'learning rate' or 'loss function'.
  
  ###### Perhaps you'd also like to visualize the distributions of activations coming off a particular layer, or the distribution of gradients or weights. Collect this data by attaching histogram_summary ops to the gradient outputs and to the variable that holds your weights, respectively.
  
  ###### Operations in TensorFlow don't do anything until you run them, or an op that depends on their output. And the summary nodes that we've just created are peripheral to your graph: none of the ops you are currently running depend on them. So, to generate summaries, we need to run all of these summary nodes. Managing them by hand would be tedious, so use tf.summary.merge_all to combine them into a single op that generates all the summary data.
  
  ###### Then, you can just run the merged summary op, which will generate a serialized Summary protobuf object with all of your summary data at a given step. Finally, to write this summary data to disk, pass the summary protobuf to a tf.summary.FileWriter.
  
  ###### The FileWriter takes a logdir in its constructor - this logdir is quite important, it's the directory where all of the events will be written out. Also, the FileWriter can optionally take a Graph in its constructor. If it receives a Graph object, then TensorBoard will visualize your graph along with tensor shape information. This will give you a much better sense of what flows through the graph: see Tensor shape information [Link: https://www.tensorflow.org/versions/master/how_tos/graph_viz/#tensor_shape_information]
  
  ###### Now that you've modified your graph and have a FileWriter, you're ready to start running your network! If you want, you could run the merged summary op every single step, and record a ton of training data. That's likely to be more data than you need, though. Instead, consider running the merged summary op every n steps.
  
  ###### The code example below is a modification of the simple MNIST tutorial, in which we have added some summary ops, and run them every ten steps. If you run this and then launch tensorboard --logdir=/tmp/mnist_logs, you'll be able to visualize statistics, such as how the weights or accuracy varied during training - https://github.com/sujayVittal/Machine-Learning-with-TensorFlow-Study-Jam-2017/blob/master/weights_accuracy_during_training-tensorboard.md
  
  ###### After we've initialized the FileWriters, we have to add summaries to the FileWriters as we train and test the model - https://github.com/sujayVittal/Machine-Learning-with-TensorFlow-Study-Jam-2017/blob/master/FileWriters-TensorBoard.md
 
 
## Execution:

##### To run TensorBoard, use the following command (alternatively python -m tensorflow.tensorboard) where logdir points to the directory where the FileWriter serialized its data. If this logdir directory contains subdirectories which contain serialized data from separate runs, then TensorBoard will visualize the data from all of those runs. Once TensorBoard is running, navigate your web browser to localhost:6006 to view the TensorBoard.:
    
    tensorboard --logdir=path/to/log-directory
    
##### When looking at TensorBoard, you will see the navigation tabs in the top right corner. Each tab represents a set of serialized data that can be visualized.

#### Play around :)
    
##### 
