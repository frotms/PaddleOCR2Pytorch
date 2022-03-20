import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import numpy as np
import cv2

import onnx

def onnx_optim(onnxfile, save_onnxfile):
    onnx_path = os.path.abspath(os.path.expanduser(onnxfile))
    save_onnxpath = os.path.abspath(os.path.expanduser(save_onnxfile))
    save_dir = os.path.dirname(save_onnxpath)
    if not os.path.exists(onnx_path):
        raise FileNotFoundError('{} is not existed.'.format(onnx))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print('makedir: {}'.format(save_dir))
    onnx_model = onnx.load(onnxfile)
    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
    # from onnx import optimizer # too old
    import onnxoptimizer
    optimized_model = onnxoptimizer.optimize(onnx_model, passes)
    optimized_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
    optimized_model.graph.input[0].type.tensor_type.shape.dim[1].dim_param = '?'
    optimized_model.graph.input[0].type.tensor_type.shape.dim[2].dim_param = '?'
    optimized_model.graph.input[0].type.tensor_type.shape.dim[3].dim_param = '?'

    optimized_model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = '?'
    optimized_model.graph.output[0].type.tensor_type.shape.dim[2].dim_param = '?'
    optimized_model.graph.output[0].type.tensor_type.shape.dim[3].dim_param = '?'

    onnx.save(optimized_model, save_onnxpath)
    print('{} is saved.'.format(save_onnxpath))


if __name__ == '__main__':
    import argparse, json, textwrap, sys, os

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', "--src_model_path", type=str, help='Assign the orginal onnx path.', default=None)
    parser.add_argument('-d', "--dst_model_path", type=str, help='Assign the saving onnx path.', default=None)
    args = parser.parse_args()

    src_model_path = args.src_model_path
    dst_model_path = args.dst_model_path
    onnx_optim(src_model_path, dst_model_path)

    print('done!')