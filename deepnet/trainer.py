import cudamat
import dbm
import dbn
import deepnet_pb2
import fastdropoutnet
import neuralnet
import sparse_coder

import json


def LockGPU():
    cudamat.cublas_init()


def FreeGPU():
    cudamat.cublas_shutdown()


def LoadExperiment(model_file, train_op_file, eval_op_file):
    model = json.load(open(model_file))
    train_op = json.load(open(train_op_file))
    eval_op = json.load(open(eval_op_file))

    return model, train_op, eval_op


def create_deepnet(model, train_op, eval_op):
    type2model = {
        "FEED_FORWARD_NET": neuralnet.NeuralNet,
        "DBM": dbm.DBM,
        "DBN": dbn.DBN,
        "SPARSE_CODER": sparse_coder.SparseCoder,
        "FAST_DROPOUT_NET": fastdropoutnet.FastDropoutNet
    }

    return type2model[model["modelType"]](model, train_op, eval_op)


def recode_to_json(proto_file, proto):
    from google.protobuf import json_format, text_format
    import os

    protoname, ext = os.path.splitext(proto_file)

    text_format.Merge(open(proto_file, 'r').read(), proto)
    res = json_format.MessageToJson(proto)

    with open(protoname + '.json', 'wb') as jout:
        jout.write(res)


def main_recode():
    folder = r'examples\ff'
    files = 'model_dropout train eval'.split(' ')
    args = [folder + '\\' + _ + '.pbtxt' for _ in files]

    recode_to_json(args[0], deepnet_pb2.Model())
    recode_to_json(args[1], deepnet_pb2.Operation())
    recode_to_json(args[2], deepnet_pb2.Operation())


def main():
    LockGPU()

    folder = r'examples\ff'
    files = 'model_dropout train eval'.split(' ')
    args = [folder + '\\' + _ + '.json' for _ in files]

    model, train_op, eval_op = LoadExperiment(*args)
    model = create_deepnet(model, train_op, eval_op)
    model.train()

    FreeGPU()

if __name__ == '__main__':
    main()
