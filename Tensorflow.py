# Simple Neural Network in Tensorflow.

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)


# Model.

x = tf.placeholder(tf.float32,[None, 784], name = "x")
W = tf.Variable(tf.zeros([784,10]), name = "W")
b = tf.Variable(tf.zeros(10), name = "b")
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32,[None,10])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = y_))
optimizer = tf.train.GradientDescentOptimizer(.5).minimize(loss)


epochs = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(epochs):
        if (epoch + 1) % 10 == 0:
            print("Epoch No.", epoch + 1,"of",epochs)
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(optimizer, feed_dict = {x : batch_xs,
                                         y_ : batch_ys})

    
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

    print("\n Accuracy:", accuracy.eval({x : mnist.test.images,
                                                   y_: mnist.test.labels}))
    
      
            
            

    
