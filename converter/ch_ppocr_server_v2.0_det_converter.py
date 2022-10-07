import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import numpy as np
import cv2
import torch
from pytorchocr.base_ocr_v20 import BaseOCRV20

class ServerV20DetConverter(BaseOCRV20):
    def __init__(self, config, paddle_pretrained_model_path, **kwargs):
        super(ServerV20DetConverter, self).__init__(config, **kwargs)
        self.load_paddle_weights(paddle_pretrained_model_path)
        self.net.eval()


    def load_paddle_weights(self, weights_path):
        import paddle.fluid as fluid
        with fluid.dygraph.guard():
            para_state_dict, opti_state_dict = fluid.load_dygraph(weights_path)

        for k,v in self.net.state_dict().items():
            keyword = 'stages.'
            if keyword in k:
                # replace: 'stages.{}.' -> ''
                start_id = k.find(keyword)
                end_id = start_id + len(keyword) + 1 + 1
                name = k.replace(k[start_id:end_id], '')
            else:
                name = k


            if name.endswith('num_batches_tracked'):
                continue

            if name.endswith('running_mean'):
                ppname = name.replace('running_mean', '_mean')
            elif name.endswith('running_var'):
                ppname = name.replace('running_var', '_variance')
            elif name.endswith('bias') or name.endswith('weight'):
                ppname = name
            else:
                print('Redundance:')
                print(name)
                raise ValueError


            self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname]))


if __name__ == '__main__':
    import argparse, json, textwrap, sys, os

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_model_path", type=str, help='Assign the paddleOCR trained model(best_accuracy)')
    args = parser.parse_args()

    cfg = {'model_type':'det',
           'algorithm':'DB',
           'Transform':None,
           'Backbone':{'name':'ResNet_vd', 'layers':18, 'disable_se':True},
           'Neck':{'name':'DBFPN', 'out_channels':256},
           'Head':{'name':'DBHead', 'k':50}}
    kwargs = {}
    paddle_pretrained_model_path = os.path.join(os.path.abspath(args.src_model_path), 'best_accuracy')
    converter = ServerV20DetConverter(cfg, paddle_pretrained_model_path, **kwargs)
    print('todo')

    # image = cv2.imread('6.jpg')
    # image = cv2.resize(image, (320, 448))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # scale = 1. / 255
    # norm_img = (image * scale - mean) / std
    # transpose_img = norm_img.transpose(2, 0, 1)
    # transpose_img = np.expand_dims(transpose_img, 0).astype(np.float32)
    # inp = torch.Tensor(transpose_img)

    # out = converter.net(inp)
    # out = out['maps'].data.numpy()
    # print('out:', np.sum(out), np.mean(out), np.max(out), np.min(out))

    # save
    converter.save_pytorch_weights('ch_ptocr_server_v2.0_det_infer.pth')
    print('done.')