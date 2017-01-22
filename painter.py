import cv2
import tensorflow as tf
import numpy as np
import numpy.random as random

''' Load the Image '''
img = cv2.imread('cat.jpg')
y_train = img.reshape(-1,3)


'''Define the required parameters'''
BATCH_SIZE = 1000
TRAIN_STEPS = 200001
INPUT_SIZE = 2
OUTPUT_SIZE = 3
HIDDEN_SIZE = [20,20,20,20,20,20,20]

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
#optimizer = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9)
train = optimizer.minimize(loss)
init = tf.initialize_all_variables()


''''A vector to draw out samples '''
x_sample = np.array([[[j,i]for i in range(img.shape[0])]for j in range(img.shape[1])])
x_sample = x_sample.reshape(-1,2) #Convert it to a column vector so that it can be fed to the NN



''' Perform Training'''
print 'Starting Training:'
with tf.Session() as sess:
    sess.run(init) #Initialize all the variables(parameters)

    for step in range(TRAIN_STEPS):
        row_batch = random.choice(img.shape[0],size=BATCH_SIZE)
        col_batch = random.choice(img.shape[1],size=BATCH_SIZE)
        x_batch = np.vstack((row_batch,col_batch)).T 
        y_batch = img[row_batch,col_batch]

        sess.run(train,feed_dict={x:x_batch,y_target:y_batch})

        if step % 50 == 0:
               losses = sess.run(loss,feed_dict={x:x_batch,y_target:y_batch})
               print 'Step: {} Loss: {}'.format(step,losses)

        #After every 1000 epochs sample an image
        if step%5000 == 0:
            y_sample = sess.run(y_pred,feed_dict={x:x_sample,y_target:y_train})
            image_sample = y_sample.reshape(img.shape[0],img.shape[1],3).astype(np.uint8)
            #Store the sample
            cv2.imwrite('samples/sample_epoch'+str(step)+'.jpg',image_sample)
            #cv2.imshow('sample_image',image_sample)
            #cv2.waitKey(0)


