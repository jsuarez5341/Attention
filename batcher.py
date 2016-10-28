import numpy as np
from pdb import set_trace as t

class Batcher():
   def __init__(self, X, Y):
      #Randomly shuffle data
      inds = np.arange(X.shape[0])
      permInds = np.random.permutation(inds)
      self.X = X[permInds]
      self.Y = Y[permInds]
      self.ind = 0

   def next(self, batchSize, movePointer=True):
      #Main batcher
      m = self.X.shape[0]
      ind = self.ind

      #Misses up to one batch. Don't care.
      if m - ind < batchSize:
         self.ind = 0
      
      nextInd = ind + batchSize
      return [self.X[ind:nextInd], self.Y[ind:nextInd]]
      if movePointer:
         self.ind = nextInd


