import torch
import torch.nn as nn
import torch.nn.functional as F

from .ppyolov2_darknet import DarkNet
from .ppyolov2_resnet import ResNet
from .ppyolov2_yolo_fpn import PPYOLOPAN
from .ppyolov2_yolo_head import YOLOv3Head

class PPYOLOv2Base(nn.Module):
    def __init__(self, **kwargs):
        super(PPYOLOv2Base, self).__init__()
        self._init_params(**kwargs)
        self._init_network()
        self._initialize_weights()

    def _init_params(self, **kwargs):
        self.num_classes = kwargs.get('INIT_num_classes', 80)
        self.arch = kwargs.get('INIT_arch', 50)
        self.scale_x_y = kwargs.get('INIT_scale_x_y', 1.05)
        self.downsample_ratio = kwargs.get('INIT_downsample_ratio', 32)
        self.anchors = [
            # [8, 9], [10, 23], [19, 15],
            # [23, 33], [40, 25], [54, 50],
            # [101, 80], [139, 145], [253, 224]

            [10, 13], [16, 30], [33, 23],
            [30, 61], [62, 45], [59, 119],
            [116, 90], [156, 198], [373, 326]
        ]
        self.anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

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

    def _init_network(self):
        if self.arch == 50:
            self._init_network_resnet50()
        elif self.arch == 101:
            self._init_network_resnet101()
        else:
            raise ValueError('INIT_arch must be [50, 101], but got {}'.format(self.arch))


    def _init_network_resnet50(self):
        self.backbone = ResNet(
            depth=50,
            ch_in=64,
            variant='d',
            lr_mult_list=[1.0, 1.0, 1.0, 1.0],
            groups=1,
            base_width=64,
            norm_type='bn',
            norm_decay=0,
            freeze_norm=False,
            freeze_at=-1,
            return_idx=[1, 2, 3],
            dcn_v2_stages=[3],
            num_stages=4,
            std_senet=False
        )

        self.neck = PPYOLOPAN(
            in_channels=[512, 1024, 2048],
            norm_type='bn',
            data_format='NCHW',
            act='mish',
            conv_block_num=3,
            drop_block=True,
            block_size=3,
            keep_prob=1.0,
            spp=True,
        )

        self.head = YOLOv3Head(
            in_channels=[1024, 512, 256],
            anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                     [59, 119], [116, 90], [156, 198], [373, 326]],
            anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
            num_classes=self.num_classes,
            loss='YOLOv3Loss',
            iou_aware=True,
            iou_aware_factor=0.5,
            data_format='NCHW'
        )

    def _init_network_resnet101(self):
        self.backbone = ResNet(
            depth=101,
            ch_in=64,
            variant='d',
            lr_mult_list=[1.0, 1.0, 1.0, 1.0],
            groups=1,
            base_width=64,
            norm_type='bn',
            norm_decay=0,
            freeze_norm=False,
            freeze_at=-1,
            return_idx=[1, 2, 3],
            dcn_v2_stages=[3],
            num_stages=4,
            std_senet=False
        )

        self.neck = PPYOLOPAN(
            in_channels=[512, 1024, 2048],
            norm_type='bn',
            data_format='NCHW',
            act='mish',
            conv_block_num=3,
            drop_block=False,
            block_size=3,
            keep_prob=1.0,
            spp=True,
        )

        self.head = YOLOv3Head(
            in_channels=[1024, 512, 256],
            anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                     [59, 119], [116, 90], [156, 198], [373, 326]],
            anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
            num_classes=self.num_classes,
            loss='YOLOv3Loss',
            iou_aware=True,
            iou_aware_factor=0.5,
            data_format='NCHW'
        )


    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def load_paddle_weights(self, weights_path):
        print('paddle weights loading...')
        import paddle.fluid as fluid
        with fluid.dygraph.guard():
            para_state_dict, opti_state_dict = fluid.load_dygraph(weights_path)

        sd = para_state_dict
        for key, value in sd.items():
            print('paddle: {} ---- {}'.format(key, value.shape))

        for key, value in self.state_dict().items():
            print('pytorch: {} ---- {}'.format(key, value.shape))

        for key, value in self.state_dict().items():

            if key.endswith('num_batches_tracked'):
                continue

            ppname = key
            ppname = ppname.replace('.running_mean', '._mean')
            ppname = ppname.replace('.running_var', '._variance')

            if key.startswith('backbone.conv'):
                pass

            if key.startswith('backbone.res_layers'):
                ppname = ppname.replace('.res_layers', '')
                ppname = ppname.replace('.blocks', '')

            if key.startswith('neck.fpn_blocks'):
                ppname = ppname.replace('.fpn_blocks', '')
                ppname = ppname.replace('.fpn_', '.fpn.')
                ppname = ppname.replace('.conv_module.0_0', '.conv_module.0.0')
                ppname = ppname.replace('.conv_module.0_1', '.conv_module.0.1')
                ppname = ppname.replace('.conv_module.1_0', '.conv_module.1.0')
                ppname = ppname.replace('.conv_module.1_1', '.conv_module.1.1')
                ppname = ppname.replace('.conv_module.2_0', '.conv_module.2.0')
                ppname = ppname.replace('.conv_module.2_1', '.conv_module.2.1')

            if key.startswith('neck.fpn_routes'):
                ppname = ppname.replace('.fpn_routes', '')
                ppname = ppname.replace('.fpn_transition_', '.fpn_transition.')

            if key.startswith('neck.pan_blocks'):
                ppname = ppname.replace('.pan_blocks', '')
                ppname = ppname.replace('.pan_', '.pan.')
                ppname = ppname.replace('.conv_module.0_0', '.conv_module.0.0')
                ppname = ppname.replace('.conv_module.0_1', '.conv_module.0.1')
                ppname = ppname.replace('.conv_module.1_0', '.conv_module.1.0')
                ppname = ppname.replace('.conv_module.1_1', '.conv_module.1.1')
                ppname = ppname.replace('.conv_module.2_0', '.conv_module.2.0')
                ppname = ppname.replace('.conv_module.2_1', '.conv_module.2.1')

            if key.startswith('neck.pan_routes'):
                ppname = ppname.replace('.pan_routes', '')
                ppname = ppname.replace('.pan_transition_', '.pan_transition.')

            if key.startswith('head.yolo_outputs'):
                ppname = ppname.replace('head.yolo_outputs.yolo_output_', 'yolo_head.yolo_output.')

            try:
                weights = sd[ppname]
                self.state_dict()[key].copy_(torch.Tensor(weights))

            except Exception as e:
                print('pp: ', key, sd[ppname].shape)
                print('pt: ', key, self.state_dict()[key].shape)
                raise e
        print('model is loaded: {}'.format(weights_path))


    def load_pytorch_weights(self, weights_path):
        self.load_state_dict(torch.load(weights_path))
        print('model is loaded: {}'.format(weights_path))


    def save_pytorch_weights(self, weights_path):
        try:
            torch.save(self.state_dict(), weights_path, _use_new_zipfile_serialization=False)
        except:
            torch.save(self.state_dict(), weights_path)  # _use_new_zipfile_serialization=False for torch>=1.6.0
        print('model is saved: {}'.format(weights_path))