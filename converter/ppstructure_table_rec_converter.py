# https://zhuanlan.zhihu.com/p/335753926
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import numpy as np
import cv2
import torch
from pytorchocr.base_ocr_v20 import BaseOCRV20

class PPStructureTableRecConverter(BaseOCRV20):
    def __init__(self, config, paddle_pretrained_model_path, **kwargs):
        para_state_dict, opti_state_dict = self.read_paddle_weights(paddle_pretrained_model_path)
        out_channels = list(para_state_dict.values())[-1].shape[0]
        print('out_channels: ', out_channels)
        print(type(kwargs), kwargs)
        kwargs['out_channels'] = out_channels
        super(PPStructureTableRecConverter, self).__init__(config, **kwargs)
        # self.load_paddle_weights(paddle_pretrained_model_path)
        self.load_paddle_weights([para_state_dict, opti_state_dict])
        print('model is loaded: {}'.format(paddle_pretrained_model_path))
        self.net.eval()


    def load_paddle_weights(self, paddle_weights):
        para_state_dict, opti_state_dict = paddle_weights
        [print('paddle: {} ---- {}'.format(k, v.shape)) for k,v in para_state_dict.items()]
        [print('pytorch: {} ---- {}'.format(k, v.shape)) for k,v in self.net.state_dict().items()]

        for k,v in self.net.state_dict().items():
            keyword = 'block_list.'
            if keyword in k:
                # replace: block_list.
                name = k.replace(keyword, '')
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
            elif 'lstm' in name:
                ppname = name

            else:
                print('Redundance:')
                print(name)
                raise ValueError

            try:
                if ppname.endswith('fc.weight'):
                    self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname].T))
                else:
                    self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname]))
            except Exception as e:
                print('pytorch: {}, {}'.format(k, v.size()))
                print('paddle: {}, {}'.format(ppname, para_state_dict[ppname].shape))
                raise e

        print('model is loaded.')


if __name__ == '__main__':
    import argparse, json, textwrap, sys, os

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_model_path", type=str, help='Assign the paddleOCR trained model(best_accuracy)')
    args = parser.parse_args()

    cfg = {'model_type':'rec',
           'algorithm':'CRNN',
           'Transform':None,
           'Backbone':{'model_name':'large', 'name':'MobileNetV3', },
           'Neck':{'name':'SequenceEncoder', 'hidden_size':96, 'encoder_type':'rnn'},
           'Head':{'name':'CTCHead', 'fc_decay': 4e-05}}
    paddle_pretrained_model_path = os.path.join(os.path.abspath(args.src_model_path), 'best_accuracy')
    converter = PPStructureTableRecConverter(cfg, paddle_pretrained_model_path)

    np.random.seed(666)
    inp = torch.from_numpy(np.random.randn(1, 3, 32, 320).astype(np.float32))
    with torch.no_grad():
        out = converter.net(inp).cpu().numpy()
    print('out:', np.sum(out), np.mean(out), np.max(out), np.min(out))

    # save
    converter.save_pytorch_weights('en_ptocr_mobile_v2.0_table_rec_infer.pth')
    print('done.')