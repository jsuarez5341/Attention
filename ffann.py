import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from pdb import set_trace as t


#Tensorflow setup
def tfInit():
   init = tf.initialize_all_variables()
   sess = tf.Session()
   sess.run(init)
   return sess

#Load MNIST
def loadDat():
   mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

   #Train
   trainDat = mnist.train

   #Val
   valDat  = mnist.validation

   #Test
   testDatX = mnist.test.images
   testDatY = mnist.test.labels

   #Dimensionality of data and num classes
   D = trainDat.images.shape[1]
   C = trainDat.labels.shape[1]

   return trainDat, valDat, testDatX, testDatY, D, C

#Setup TF symbolics
def initParams(nil):
   global x, W, b, y, a
   D = nil[0]
   C = nil[-1]
   x = tf.placeholder(tf.float32, [None, D])
   W = []
   for l in range(1, len(nil)):
      W += [tf.Variable(tf.random_normal(([nil[l-1], nil[l]]), stddev=1e-2))]
   y = tf.placeholder(tf.float32, [None, C])
   a = pred()
   
#Forward pass
def pred():
   global x, W
   a = x
   for l in range(len(W)-1):
      a = tf.nn.relu(tf.matmul(a, W[l]))
   return tf.nn.softmax(tf.matmul(a, W[-1]))

#Network cost
def loss():
   global y, a
   return tf.reduce_mean(-tf.reduce_sum(y * tf.log(a), reduction_indices=[1]))

#Generic gradient based optimizer
def optimizer(eta=0.5):
   return tf.train.GradientDescentOptimizer(eta).minimize(loss())

def train(trainDat, valDat):
   for i in range(maxIters):
      if i % epoch == 0:
         xTrain, yTrain = trainDat.next_batch(valBatchSize)
         acc = test(xTrain, yTrain)
         print 'Training accuracy at iteration ' + str(i) + ': ' + str(acc)
         xVal, yVal = valDat.next_batch(valBatchSize)
         acc = test(xVal, yVal)
         print 'Validation accuracy at iteration ' + str(i) + ': ' + str(acc)
         
      #Minibatch inputs
      xBatch, yBatch = trainDat.next_batch(batchSize)
      sess.run(opt, feed_dict={x:xBatch, y:yBatch})

def test(testX, testY):
   global y, a
   predMask = tf.equal(tf.argmax(a,1), tf.argmax(y,1))
   accuracy = tf.reduce_mean(tf.cast(predMask, tf.float32))
   return sess.run(accuracy, feed_dict={x:testX, y:testY})
   

if __name__ == '__main__':
   trainDat, valDat, testX, testY, D, C = loadDat()

   #Fun with python scoping rules
   #We don't have to directly pass these as args.
   maxIters = 10000
   batchSize = 100
   valBatchSize = 500
   epoch = 100
   nil = [D, 1000, 1000, C]

   initParams(nil)
   opt = optimizer(eta=0.5)
   sess = tfInit()
   train(trainDat, valDat)
   acc = test(testX, testY)

   print acc



