import tensorflow as tf
import numpy as np
from batcher import Batcher
from pdb import set_trace as tt

#Tensorflow setup
def tfInit():
   init = tf.initialize_all_variables()
   sess = tf.Session()
   sess.run(init)
   return sess

def loadDat():
   dat = np.sin(np.linspace(0,1000,200000))
   x = dat
   x = x.reshape(-1, 10)
   y = x[1:, :-5]
   x = x[:-1]
   
   #Make train/val/test splits
   xTrain = x[:7000]
   yTrain = y[:7000]
   xVal = x[7000:8000]
   yVal = y[7000:8000]
   xTest = x[8000:]
   yTest = y[8000:]

   #Make data batchers
   trainDat = Batcher(xTrain, yTrain)
   valDat = Batcher(xVal, yVal)
   testDat = Batcher(xTest, yTest)
   return trainDat, valDat, testDat

#Setup TF symbolics
def initParams(nil):
   global x, y, a, o, s, U, W, V, sAry, window
   D = nil[0]
   H = nil[1]
   C = nil[-1]
   x = tf.placeholder(tf.float32, [None, D])
   y = tf.placeholder(tf.float32, [None, D-window])
   a = tf.placeholder(tf.float32, [None, C])
   U = tf.Variable(tf.random_normal(([window, H]), stddev=1e-2))
   W = tf.Variable(tf.random_normal(([H, H]), stddev=1e-2))
   V = tf.Variable(tf.random_normal(([H, 1]), stddev=1e-2))
   s = tf.placeholder(tf.float32, [None, H])
   sAry = np.array([])
 
#Network cost
def loss():
   o = pred()
   costs = tf.nn.softmax_cross_entropy_with_logits(o, y)
   return tf.reduce_sum(costs)

def optimizer(eta=0.5):
   return tf.train.GradientDescentOptimizer(eta).minimize(loss())

#Forward pass
def pred():
   s = tf.nn.tanh(tf.matmul(x[:, :window], U))
   T = x._shape_as_list()[1]
   o = [tf.matmul(s, V)]
   for t in range(1, T-window):
      s = tf.nn.tanh(tf.matmul(x[:, t:t+window], U) + tf.matmul(s, W))
      o += [tf.matmul(s, V)]
   o = tf.concat(1, o)
   return o

def train(trainDat, valDat):
   for i in range(maxIters):
      if i % epoch == 0:
         xTrain, yTrain = trainDat.next(valBatchSize, movePointer=False)
         acc = test(trainDat)
         print 'Training cost at iteration ' + str(i) + ': ' + str(acc)
         xVal, yVal = valDat.next(valBatchSize)
         acc = test(valDat)
         print 'Validation cost at iteration ' + str(i) + ': ' + str(acc)

      #Minibatch inputs
      xBatch, yBatch = trainDat.next(batchSize)

      sess.run(opt, feed_dict={x:xBatch, y:yBatch})

def test(testDat):
   #Regression task, just use loss.
   xTest, yTest = testDat.next(numTest)
   #sAry = sess.run(pred(), feed_dict={x:xTest, s:sAry})#, y:yTest})
   cost = sess.run(loss(), feed_dict={x:xTest, y:yTest})
   return cost

def dream(testDat):
   xTest, yTest = testDat.next(1)
   dreamtOf = sess.run(pred(), feed_dict={x:xTest})
   print dreamtOf
      
#Sample main training routine
if __name__ == '__main__':
   #Fun with python scoping rules
   #We don't have to directly pass these as args.
   maxIters = 1005
   batchSize = 100
   valBatchSize = 500
   numTest = 900
   epoch = 1000
   nil = [10, 40, 10]
   window = 5
   H = nil[1]

   #Create data batchers 
   trainDat, valDat, testDat = loadDat()

   initParams(nil)
   opt = optimizer(eta=0.001)
   sess = tfInit()
   train(trainDat, valDat)
   acc = test(testDat)
   dream(testDat)

   print acc

