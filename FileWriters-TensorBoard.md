## After we've initialized the FileWriters, we have to add summaries to the FileWriters as we train and test the model. Use the following:
  
  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}
  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  
      summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
      train_writer.add_summary(summary, i)
      
      
## Use this and try building the graph :)
