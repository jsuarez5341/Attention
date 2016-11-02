import tensorflow as tf
import numpy as np
from toyDataGen import genDat
import batcher
from pdb import set_trace as tt

#Tensorflow setup
def tfInit():
   init = tf.initialize_all_variables()
   sess = tf.Session()
   sess.run(init)
   return sess

def initParams():
   global deeplstm, W, b, o, x, y, s
   lstm = tf.nn.rnn_cell.BasicLSTMCell(H)
   deeplstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * numLayers)
   W = tf.Variable(tf.random_normal((2*H, 1), stddev=1e-2))
   b = tf.Variable(tf.random_normal((batchSize, 1), stddev=1e-2))
   o = tf.zeros([batchSize, H])
   
   x = tf.placeholder(tf.float32, [batchSize, T])
   y = tf.placeholder(tf.float32, [batchSize, T])
   s = tf.placeholder(tf.float32, [batchSize, deeplstm.state_size])
   #s = tf.zeros([batchSize, lstm.state_size])


   #embeddings = tf.Variable(
   #   tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))



def loss(logits, y):
   return tf.reduce_sum(tf.square(y-logits)) / batchSize
   return tf.nn.softmax_cross_entropy_with_logits(logits, y)  


def score(hDst, hSrc):
   return tf.reduce_sum(hDst * hSrc, 0)
   return tf.dot(hDst, hSrc)

def align(hSrcAry, hDst):
   denom = 0
   for e in hSrcAry:
      denom += tf.exp(score(hDst, e))
   
   out = []
   for e in hSrcAry:
      out += [tf.exp(score(hDst, e)) / denom]
   out = tf.transpose(tf.pack(out))
   return out

def getContext(hSrcAry, alph):
   ciVec = 0
   N = len(hSrcAry)
   for i in range(N):
      ciVec += (1.0/N) * alph[:, i] * hSrcAry[i]
   return ciVec
   
   
def encodeDecode():
   global deeplstm, W, b, o, x, y, s, embeddings

   #embed = tf.nn.embedding_lookup(embeddings, x)

   sNew = s
   err = 0.0
   lenSrc = y._shape_as_list()[1] 
   hSAry = []
   for t in range(lenSrc):
      if t > 0:
         tf.get_variable_scope().reuse_variables()

      #LSTM Timestep activation
      o, sNew = deeplstm(x[:,t:t+1], sNew)
      hSAry += [o]

   lenDst = y._shape_as_list()[1] 
   preds = []
   for t in range(lenDst):

      #LSTM Timestep activation
      o, sNew = deeplstm(x[:,t:t+1], sNew)

      #o = hd. Implement attention
      alph = align(hSAry, o)
      c = getContext(hSAry, alph)
      oc = tf.concat(1, [o,c])
   
      #Compute predictions and loss
      logits = tf.matmul(oc, W) + b
      preds += [logits]
      err += loss(logits, y[:,t:t+1])
   preds = tf.concat(1, preds)
   return err, preds 


def pred():
   global deeplstm, W, b, o, x, s

   sNew = s
   #LSTM Timestep activation
   o, sNew = deeplstm(x[:,0:1], sNew)

   #Compute predictions and loss
   logits = tf.matmul(o, W) + b
   #probs.append(tf.nn.softmax(logits))
   tf.get_variable_scope().reuse_variables()
   return logits

#Get some notion of error
def test(batchGen, batchSize):
   err = 0
   for i in range(10): 
      print i
      xBatch, yBatch = batchGen.next(batchSize)
      sNp = np.zeros((batchSize, deeplstm.state_size), np.float32)
      deltaErr, sPred = sess.run(encodeDecode(), feed_dict={x:xBatch, y:yBatch, s:sNp})
      err += deltaErr
   return err 

def optimizer(err, eta=0.5):
   return tf.train.AdamOptimizer().minimize(err)

def getData(m, T):
   xTrain, yTrain = genDat(m, T)
   batchGen = batcher.Batcher(xTrain, yTrain)
   return batchGen

def train(trainGen, batchSize, maxIters=100):
   for i in range(maxIters):
      if i % 10 == 0:
         print i
      xBatch, yBatch = trainGen.next(batchSize)
      sNp = np.zeros((batchSize, deeplstm.state_size), np.float32)
      sess.run(opt, feed_dict={x:xBatch, y:yBatch, s:sNp})
    
if __name__ == '__main__':
   m = 2
   T = 2
   H = 50
   batchSize = 2 
   maxIters = 500
   numLayers = 2

   batchGen = getData(m, T)
   initParams()
   #err, ss = seqForward()
   err, ss = encodeDecode()
   opt = optimizer(err, eta=0.1)
   sess = tfInit()
   train(batchGen, batchSize, maxIters)

   xTest, yTest = batchGen.next(batchSize)
   sNp = np.zeros((batchSize, deeplstm.state_size), np.float32)
   print xTest
   print yTest
   print sess.run(encodeDecode(), feed_dict={x:xTest, y:yTest, s:sNp})

   






