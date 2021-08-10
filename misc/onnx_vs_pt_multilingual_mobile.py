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



def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 if_act=True,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)

        self.bn = nn.BatchNorm2d(
            out_channels,
            )
        if self.if_act:
            self.act = Activation(act_type=act, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            x = self.act(x)
        return x

class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=4, name=""):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.relu1 = Activation(act_type='relu', inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.hard_sigmoid = Activation(act_type='hard_sigmoid', inplace=True)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = self.relu1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.hard_sigmoid(outputs)
        outputs = inputs * outputs
        return outputs

class ResidualUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 use_se,
                 act=None,
                 name=''):
        super(ResidualUnit, self).__init__()
        self.if_shortcut = stride == 1 and in_channels == out_channels
        self.if_se = use_se

        self.expand_conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act=act,
            name=name + "_expand")
        self.bottleneck_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=int((kernel_size - 1) // 2),
            groups=mid_channels,
            if_act=True,
            act=act,
            name=name + "_depthwise")
        if self.if_se:
            self.mid_se = SEModule(mid_channels, name=name + "_se")
        self.linear_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None,
            name=name + "_linear")

    def forward(self, inputs):
        x = self.expand_conv(inputs)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = inputs + x
        return x

class MobileNetV3(nn.Module):
    def __init__(self,
                 in_channels=3,
                 model_name='small',
                 scale=0.5,
                 large_stride=None,
                 small_stride=None,
                 **kwargs):
        super(MobileNetV3, self).__init__()
        if small_stride is None:
            small_stride = [2, 2, 2, 2]
        if large_stride is None:
            large_stride = [1, 2, 2, 2]

        assert isinstance(large_stride, list), "large_stride type must " \
                                               "be list but got {}".format(type(large_stride))
        assert isinstance(small_stride, list), "small_stride type must " \
                                               "be list but got {}".format(type(small_stride))
        assert len(large_stride) == 4, "large_stride length must be " \
                                       "4 but got {}".format(len(large_stride))
        assert len(small_stride) == 4, "small_stride length must be " \
                                       "4 but got {}".format(len(small_stride))

        if model_name == "large":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', large_stride[0]],
                [3, 64, 24, False, 'relu', (large_stride[1], 1)],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', (large_stride[2], 1)],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hard_swish', 1],
                [3, 200, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 480, 112, True, 'hard_swish', 1],
                [3, 672, 112, True, 'hard_swish', 1],
                [5, 672, 160, True, 'hard_swish', (large_stride[3], 1)],
                [5, 960, 160, True, 'hard_swish', 1],
                [5, 960, 160, True, 'hard_swish', 1],
            ]
            cls_ch_squeeze = 960
        elif model_name == "small":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', (small_stride[0], 1)],
                [3, 72, 24, False, 'relu', (small_stride[1], 1)],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hard_swish', (small_stride[2], 1)],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 120, 48, True, 'hard_swish', 1],
                [5, 144, 48, True, 'hard_swish', 1],
                [5, 288, 96, True, 'hard_swish', (small_stride[3], 1)],
                [5, 576, 96, True, 'hard_swish', 1],
                [5, 576, 96, True, 'hard_swish', 1],
            ]
            cls_ch_squeeze = 576
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert scale in supported_scale, \
            "supported scales are {} but input scale is {}".format(supported_scale, scale)

        inplanes = 16
        # conv1
        self.conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=make_divisible(inplanes * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            if_act=True,
            act='hard_swish',
            name='conv1')
        i = 0
        block_list = []
        inplanes = make_divisible(inplanes * scale)
        for (k, exp, c, se, nl, s) in cfg:
            block_list.append(
                ResidualUnit(
                    in_channels=inplanes,
                    mid_channels=make_divisible(scale * exp),
                    out_channels=make_divisible(scale * c),
                    kernel_size=k,
                    stride=s,
                    use_se=se,
                    act=nl,
                    name='conv' + str(i + 2)))
            inplanes = make_divisible(scale * c)
            i += 1
        self.blocks = nn.Sequential(*block_list)

        self.conv2 = ConvBNLayer(
            in_channels=inplanes,
            out_channels=make_divisible(scale * cls_ch_squeeze),
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            if_act=True,
            act='hard_swish',
            name='conv_last')

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.out_channels = make_divisible(scale * cls_ch_squeeze)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x


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
    pt_path = 'om_ug_mobile_v2.0_rec_infer.pth'
    onnx_path = 'om_ug_mobile_v2.0_rec_infer_optim.onnx'

    # mobile
    cfg = {'model_type': 'rec',
           'algorithm': 'CRNN',
           'Transform': None,
           'Backbone': {'model_name': 'small', 'name': 'MobileNetV3', 'scale': 0.5, 'small_stride': [1, 2, 2, 2]},
           'Neck': {'name': 'SequenceEncoder', 'hidden_size': 48, 'encoder_type': 'om'},
           'Head': {'name': 'CTCHead', 'fc_decay': 4e-05},
           }

    inp = np.random.randn(*input_size).astype(np.float32)
    print('input: ', inp.shape)

    onnxmodel = OnnxInfer(onnx_path)
    onnx_res = onnxmodel.run(inp.copy())
    print('onnx: ', onnx_res[0].shape)

    aux_dict = {'out_channels':onnx_res[0].shape[-1]}

    ptmodel = Crnn(cfg, **aux_dict)
    ptmodel.load_pytorch_weights(pt_path)
    ptmodel.dummy_run()
    pt_res = ptmodel.inference(torch.from_numpy(inp.copy()))
    print('pytorch: ', pt_res.shape)

    a = onnx_res[0]
    b = pt_res
    print('diff: ', np.sum(np.abs(a - b)), np.mean(np.abs(a - b)), np.max(np.abs(a - b)))
    print('done.')