import argparse

'''Parse command line arguments   '''
parser = argparse.ArgumentParser()
parser.add_argument('filename',help='Name of the image')
parser.add_argument('-hlayers',help='Hidden Layers, pass as string with numbers separated by commas')
parser.add_argument('-epoch',help='Number of epochs to train',type=int)
parser.add_argument('-savemodel',help='1 to save,default 0',type=int)
args = parser.parse_args()


import cv2
import tensorflow as tf
import numpy as np
import numpy.random as random
import os



''' Load the Image '''
filename = args.filename
img = cv2.imread(filename)
img = cv2.resize(img,(225,225))
y_train = img.reshape(-1,3)
y_sample = y_train.copy()


'''Define the required parameters'''
BATCH_SIZE = 1000
TRAIN_STEPS = int(4000*y_train.shape[0]/BATCH_SIZE)
if args.epoch:
    TRAIN_STEPS = int(args.epoch*y_train.shape[0]/BATCH_SIZE)
INPUT_SIZE = 2
OUTPUT_SIZE = 3
HIDDEN_SIZE = [20,20,20,20,20,20,20]
if args.hlayers:
    HIDDEN_SIZE = map(int,args.hlayers.split(','))

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
loss = tf.reduce_mean(tf.square(y_pred-y_target))

optimizer = tf.train.AdamOptimizer()
#optimizer = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9)
train = optimizer.minimize(loss)
init = tf.initialize_all_variables()


''''A vector to draw out samples '''
x_sample = np.array([[[j,i]for i in range(img.shape[0])]for j in range(img.shape[1])])
x_sample = x_sample.reshape(-1,2) #Convert it to a column vector so that it can be fed to the NN

x_train = x_sample.copy()
#Random permute training data
idx = random.permutation(x_train.shape[0])
x_train = x_train[idx]
y_train = y_train[idx]


''' Perform Training'''
print '\n \n \n'
print 'The Network Architecture is an FCNET with: {} layers'.format(layers)
print 'Image being painted: {}'.format(filename)
print 'Press enter to start training:'
raw_input()
print 'Starting Training:'
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init) #Initialize all the variables(parameters)
    sample_point = 0
    epoch = 0
    for step in range(TRAIN_STEPS):
        if sample_point+BATCH_SIZE < y_train.shape[0]:
            x_batch = x_train[sample_point:sample_point+BATCH_SIZE]
            y_batch = y_train[sample_point:sample_point+BATCH_SIZE]
            sample_point += BATCH_SIZE
        else:
            x_batch = x_train[BATCH_SIZE:]
            y_batch = y_train[BATCH_SIZE:]
            sample_point = 0


        sess.run(train,feed_dict={x:x_batch,y_target:y_batch})

        if step % 50 == 0:
               losses = sess.run(loss,feed_dict={x:x_batch,y_target:y_batch})
               print 'Step: {} Loss: {}'.format(step,losses)

        #After 100 every epoch sample an image
        if sample_point == 0:
            epoch += 1
            if epoch%100 == 0:
                y_sample = sess.run(y_pred,feed_dict={x:x_sample,y_target:y_train})
                image_sample = y_sample.reshape(img.shape[0],img.shape[1],3).astype(np.uint8)
                #Store the sample
                sample_filename = 'samples/sample_'+os.path.splitext(filename)[0]+'_epoch'+str(epoch)+'.jpg'
                print 'Sample stored in:  '+sample_filename
                cv2.imwrite(sample_filename,image_sample)
    if args.savemodel == 1:
        saver.save(sess,'models/model_'+os.path.splitext(filename)[0]+'_'+str(img.shape[0])+'x'+str(img.shape[1]),global_step=step)

