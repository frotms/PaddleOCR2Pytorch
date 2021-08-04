import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Crnn"]

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(1.2 * x + 3., inplace=self.inplace) / 6.

class Activation(nn.Module):
    def __init__(self, act_type, inplace=True):
        super(Activation, self).__init__()
        act_type = act_type.lower()
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=inplace)
        elif act_type == 'relu6':
            self.act = nn.ReLU6(inplace=inplace)
        elif act_type == 'sigmoid':
            raise NotImplementedError
        elif act_type == 'hard_sigmoid':
            self.act = Hsigmoid(inplace)
        elif act_type == 'hard_swish':
            self.act = Hswish(inplace=inplace)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(inplace=inplace)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        return self.act(inputs)


class ConvBNLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            groups=1,
            is_vd_mode=False,
            act=None,
            name=None, ):
        super(ConvBNLayer, self).__init__()
        self.act = act
        self.is_vd_mode = is_vd_mode
        self._pool2d_avg = nn.AvgPool2d(
            kernel_size=stride, stride=stride, padding=0, ceil_mode=True)

        self._conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1 if is_vd_mode else stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False)

        self._batch_norm = nn.BatchNorm2d(
            out_channels,)
        if self.act is not None:
            self._act = Activation(act_type=act, inplace=True)

    def forward(self, inputs):
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if self.act is not None:
            y = self._act(y)
        return y


class BottleneckBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 name=None):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act='relu',
            name=name + "_branch2a")
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2b")
        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None,
            name=name + "_branch2c")

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=stride,
                is_vd_mode=not if_first and stride[0] != 1,
                name=name + "_branch1")

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)

        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = short + conv2
        y = F.relu(y)
        return y


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 name=None):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2a")
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            act=None,
            name=name + "_branch2b")

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                is_vd_mode=not if_first and stride[0] != 1,
                name=name + "_branch1")

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = short + conv1
        y = F.relu(y)
        return y


class ResNet(nn.Module):
    def __init__(self, in_channels=3, layers=50, **kwargs):
        super(ResNet, self).__init__()

        self.layers = layers
        supported_layers = [18, 34, 50, 101, 152, 200]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_channels = [64, 256, 512,
                        1024] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        self.conv1_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            act='relu',
            name="conv1_1")
        self.conv1_2 = ConvBNLayer(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            act='relu',
            name="conv1_2")
        self.conv1_3 = ConvBNLayer(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            act='relu',
            name="conv1_3")
        self.pool2d_max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block_list = nn.Sequential()
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                conv_name = "res" + str(block + 2) + chr(97 + i)
                if i == 0 and block != 0:
                    stride = (2, 1)
                else:
                    stride = (1, 1)

                basic_block = BasicBlock(in_channels=num_channels[block] if i == 0 else num_filters[block],
                                         out_channels=num_filters[block],
                                         stride=stride,
                                         shortcut=shortcut,
                                         if_first=block == i == 0,
                                         name=conv_name)

                shortcut = True
                self.block_list.add_module('bb_%d_%d' % (block, i), basic_block)
            self.out_channels = num_filters[block]
        self.out_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, inputs):
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        y = self.pool2d_max(y)
        for block in self.block_list:
            y = block(y)
        y = self.out_pool(y)

        return y


class Im2Seq(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        x = x.squeeze(dim=2)
        x = x.permute(0,2,1)
        return x


class EncoderWithRNN_(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithRNN_, self).__init__()
        self.out_channels = hidden_size * 2
        self.rnn1 = nn.LSTM(in_channels, hidden_size, bidirectional=False, batch_first=True, num_layers=2)
        self.rnn2 = nn.LSTM(in_channels, hidden_size, bidirectional=False, batch_first=True, num_layers=2)

    def forward(self, x):
        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()
        out1, h1 = self.rnn1(x)
        out2, h2 = self.rnn2(torch.flip(x, [1]))
        return torch.cat([out1, torch.flip(out2, [1])], 2)


class EncoderWithRNN_org(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithRNN_org, self).__init__()
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(
            in_channels, hidden_size, num_layers=2, batch_first=True, bidirectional=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x

class EncoderWithRNN(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithRNN, self).__init__()
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(
            in_channels, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(
            in_channels, hidden_size, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        x, _ = self.lstm2(x)
        return x

class EncoderWithRNN_StackLSTM(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithRNN_StackLSTM, self).__init__()
        self.out_channels = hidden_size * 2
        self.lstm_0_cell_fw = nn.LSTM(in_channels, hidden_size, bidirectional=False, batch_first=True, num_layers=1)
        self.lstm_0_cell_bw = nn.LSTM(in_channels, hidden_size, bidirectional=False, batch_first=True, num_layers=1)
        self.lstm_1_cell_fw = nn.LSTM(self.out_channels, hidden_size, bidirectional=False, batch_first=True, num_layers=1)
        self.lstm_1_cell_bw = nn.LSTM(self.out_channels, hidden_size, bidirectional=False, batch_first=True, num_layers=1)

    def bi_lstm(self, x, fw_fn, bw_fn):
        out1, h1 = fw_fn(x)
        out2, h2 = bw_fn(torch.flip(x, [1]))
        return torch.cat([out1, torch.flip(out2, [1])], 2)

    def forward(self, x):
        x = self.bi_lstm(x, self.lstm_0_cell_fw, self.lstm_0_cell_bw)
        x = self.bi_lstm(x, self.lstm_1_cell_fw, self.lstm_1_cell_bw)
        return x


class EncoderWithFC(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithFC, self).__init__()
        self.out_channels = hidden_size
        self.fc = nn.Linear(
            in_channels,
            hidden_size,
            bias=True,
            )

    def forward(self, x):
        x = self.fc(x)
        return x


class SequenceEncoder(nn.Module):
    def __init__(self, in_channels, encoder_type, hidden_size=48, **kwargs):
        super(SequenceEncoder, self).__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        if encoder_type == 'reshape':
            self.only_reshape = True
        else:
            support_encoder_dict = {
                'reshape': Im2Seq,
                'fc': EncoderWithFC,
                'rnn': EncoderWithRNN,
                'om': EncoderWithRNN_StackLSTM
            }
            assert encoder_type in support_encoder_dict, '{} must in {}'.format(
                encoder_type, support_encoder_dict.keys())

            self.encoder = support_encoder_dict[encoder_type](
                self.encoder_reshape.out_channels, hidden_size)
            self.out_channels = self.encoder.out_channels
            self.only_reshape = False

    def forward(self, x):
        x = self.encoder_reshape(x)
        if not self.only_reshape:
            x = self.encoder(x)
        return x


class CTCHead(nn.Module):
    def __init__(self, in_channels, out_channels=6625, fc_decay=0.0004, **kwargs):
        super(CTCHead, self).__init__()
        self.fc = nn.Linear(
            in_channels,
            out_channels,
            bias=True,)
        self.out_channels = out_channels

    def forward(self, x, labels=None):
        predicts = self.fc(x)
        if not self.training:
            predicts = F.softmax(predicts, dim=2)
        return predicts

def build_backbone(config):
    module_name = config.pop('name')
    module_class = eval(module_name)(**config)
    return module_class

def build_neck(config):
    module_name = config.pop('name')
    module_class = eval(module_name)(**config)
    return module_class

def build_head(config, **kwargs):
    module_name = config.pop('name')
    module_class = eval(module_name)(**config, **kwargs)
    return module_class

class BaseModel(nn.Module):
    def __init__(self, config, **kwargs):
        """
        args:
            config (dict): the super parameters for module.
        """
        super(BaseModel, self).__init__()

        in_channels = config.get('in_channels', 3)

        config["Backbone"]['in_channels'] = in_channels
        self.backbone = build_backbone(config["Backbone"])
        in_channels = self.backbone.out_channels

        if 'Neck' not in config or config['Neck'] is None:
            self.use_neck = False
        else:
            self.use_neck = True
            config['Neck']['in_channels'] = in_channels
            self.neck = build_neck(config['Neck'])
            in_channels = self.neck.out_channels

        config["Head"]['in_channels'] = in_channels
        self.head = build_head(config["Head"], **kwargs)

        self._initialize_weights()

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):
        x = self.backbone(x)
        if self.use_neck:
            x = self.neck(x)
        x = self.head(x)
        return x

class Crnn:
    def __init__(self, config, **kwargs):
        print(config)
        self.config = config
        use_gpu = kwargs.get('USE_GPU', True)
        self.use_gpu = torch.cuda.is_available() and use_gpu
        self.build_net(**kwargs)
        self.net.eval()
        if self.use_gpu:
            self.net.cuda()
        self.dummy_run()

    def build_net(self, **kwargs):
        self.net = BaseModel(self.config, **kwargs)

    def inference(self, inputs):
        with torch.no_grad():
            if self.use_gpu:
                inputs = inputs.cuda()
            infer = self.net(inputs)
        return infer.cpu().numpy()

    def dummy_run(self, input_size=(1,3,32,320)):
        inp = np.random.randn(*input_size).astype(np.float32)
        inputs = torch.from_numpy(inp)
        with torch.no_grad():
            if self.use_gpu:
                inputs = inputs.cuda()
            _ = self.net(inputs)
        print('dummy run is done.')

    def export_onnx(self, onnx_path='crnn.onnx', input_size=(1,3,32,320), opset=11):
        dummy_input = torch.autograd.Variable(torch.randn(*input_size))
        torch.onnx.export(self.net, dummy_input, onnx_path, opset_version=opset,
                          do_constant_folding=False, verbose=False)
        print('{} is saved.'.format(onnx_path))

    def save_pytorch_weights(self, weights_path):
        try:
            torch.save(self.net.state_dict(), weights_path, _use_new_zipfile_serialization=False)
        except:
            torch.save(self.net.state_dict(), weights_path) # _use_new_zipfile_serialization=False for torch>=1.6.0
        print('model is saved: {}'.format(weights_path))

    def load_pytorch_weights(self, weights_path):
        self.net.load_state_dict(torch.load(weights_path))
        self.net.eval()
        print('model is loaded: {}'.format(weights_path))


import onnxruntime as rt
class OnnxInfer:
    def __init__(self, onnx_path):
        self.model_path = os.path.abspath(os.path.expanduser(onnx_path))
        if not os.path.exists(self.model_path):
            raise FileNotFoundError('{} is not existed.'.format(self.model_path))

        self.sess = rt.InferenceSession(self.model_path)
        self.input_name = self.get_input_name(self.sess)
        self.output_name = self.get_output_name(self.sess)
        print('{} is loaded.'.format(self.model_path))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def run(self, image_numpy):
        '''
        # image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: iimage_numpy})
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        out = self.sess.run(self.output_name, input_feed=input_feed)
        return out


if __name__ == '__main__':
    print('begin..')
    np.random.seed(666)
    input_size = (1, 3, 32, 320)
    pt_path = 'om_ch_ptocr_server_v2.0_rec_infer.pth'
    onnx_path = 'ch_ptocr_server_v2.0_rec_infer_optim.onnx'

    # server
    cfg = {
        'Backbone': {'name': 'ResNet', 'layers': 34},
        'Neck': {'name': 'SequenceEncoder', 'hidden_size': 256, 'encoder_type': 'om'},
        'Head': {'name': 'CTCHead', 'fc_decay': 4e-05}
    }

    inp = np.random.randn(*input_size).astype(np.float32)
    print('input: ', inp.shape)
    ptmodel = Crnn(cfg)
    ptmodel.load_pytorch_weights(pt_path)
    ptmodel.dummy_run()
    pt_res = ptmodel.inference(torch.from_numpy(inp.copy()))
    print(pt_res.shape)

    onnxmodel = OnnxInfer(onnx_path)
    onnx_res = onnxmodel.run(inp.copy())
    print(onnx_res[0].shape)
    a = pt_res
    b = onnx_res[0]
    print('diff: ', np.sum(np.abs(a - b)), np.mean(np.abs(a - b)), np.max(np.abs(a - b)))
    print('done.')