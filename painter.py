import cv2
import tensorflow as tf
import numpy as np

''' Load the Image '''
img = cv2.imread('itachi_uchiha.jpg')
x_train = cv2.resize(img,(400,400))
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',x_train)
cv2.waitKey(0)


'''Define the required parameters'''
BATCH_SIZE = 10
TRAIN_STEPS = 10000
INPUT_SIZE = 2
OUTPUT_SIZE = 3
HIDDEN_SIZE = [20,20,20]

'''Define the Model'''
x = tf.placeholder(tf.float32,shape=[None,INPUT_SIZE])
y_target = tf.placeholder(tf.float32,shape=[None,OUTPUT_SIZE])

'''Form the network architecture '''
W = {} 
b = {}
z = {}
z[0] = x
layers = list()
layers.append(INPUT_SIZE)
layers.extend(HIDDEN_SIZE)
layers.append(OUTPUT_SIZE)

for i in range(1,len(layers)-1):
    W[i] = tf.Variable(tf.random_uniform([layers[i-1],layers[i]],minval=0,maxval=0.1))
    b[i] = tf.Variable(tf.zeros([layers[i]]))
    z[i] = tf.nn.relu(tf.matmul(z[i-1],W[i])+b[i])

#Final Regression Layer
W[i+1] = tf.Variable(tf.random_uniform([layers[i],layers[i+1]],minval=0,maxval=0.1))
b[i+1] = tf.Variable(tf.zeros([layers[i+1]]))
y_pred = tf.matmul(z[i],W[i+1]) + b[i+1]

#L2 Regression loss
loss = tf.reduce_mean(tf.nn.l2_loss(y_pred-y_target))

optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)
init = tf.initialize_all_variables()


''' Perform Training'''
print 'Starting Training:'
with tf.Session() as sess:
    sess.run(init) #Initialize all the variables(parameters)

    #for step in range(TRAIN_STEPS):
        


