from neuralnet import *
from fastdropoutnet import *
from dbm import *
from dbn import *
from sparse_coder import *
from choose_matrix_library import *
import numpy as np
from time import sleep

def LockGPU():
  cm.cublas_init()

def FreeGPU():
  cm.cublas_shutdown()

def LoadExperiment(model_file, train_op_file, eval_op_file):
  model = util.ReadModel(model_file)
  train_op = util.ReadOperation(train_op_file)
  eval_op = util.ReadOperation(eval_op_file)
  return model, train_op, eval_op

def CreateDeepnet(model, train_op, eval_op):
  if model.model_type == deepnet_pb2.Model.FEED_FORWARD_NET:
    return NeuralNet(model, train_op, eval_op)
  elif model.model_type == deepnet_pb2.Model.DBM:
    return DBM(model, train_op, eval_op)
  elif model.model_type == deepnet_pb2.Model.DBN:
    return DBN(model, train_op, eval_op)
  elif model.model_type == deepnet_pb2.Model.SPARSE_CODER:
    return SparseCoder(model, train_op, eval_op)
  elif model.model_type == deepnet_pb2.Model.FAST_DROPOUT_NET:
    return FastDropoutNet(model, train_op, eval_op)
  else:
    raise Exception('Model not implemented.')

# if __name__ == '__main__':

LockGPU()

folder = r'examples\ff'
files = 'model_dropout train eval'.split(' ')
args = [folder + '\\' + _ + '.pbtxt' for _ in files]

model, train_op, eval_op = LoadExperiment(*args)

model = CreateDeepnet(model, train_op, eval_op)

model.Train()

model.edge[0].hyperparams.enable_display = True
model.edge[0].Show()
plt.show()

# FreeGPU()

