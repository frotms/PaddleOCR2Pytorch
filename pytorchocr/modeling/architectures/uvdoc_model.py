# Copyright (c) 2024 PytorchOCR Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
UVDoc: Neural Grid-based Document Unwarping (CGU-Net)

Paper: https://arxiv.org/abs/2302.02887
Original repo: https://github.com/tanguymagne/UVDoc

This is a PyTorch port of the PaddlePaddle UVDoc model used in PaddleOCR v3.x
for document image rectification/unwarping.

The model is a fully convolutional dual-task network that predicts:
1. A 2D unwarping grid (HxWx2) — backward mapping coordinates
2. A 3D grid mesh (HxWx3) — document 3D shape

Weight keys are matched to PaddlePaddle's state_dict for direct conversion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableScale(nn.Module):
    """Matches Paddle's LearnableAffineBlock with a single _weight parameter."""
    def __init__(self):
        super().__init__()
        self._weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * self._weight


class UVDocModel(nn.Module):
    """
    UVDoc CGU-Net for document image unwarping.

    Input: [B, 3, 488, 712] (the standard UVDoc input size)
    Output: dict with 'unwarp_grid' [B, 2, 31, 45] and 'mesh_3d' [B, 3, 31, 45]

    The model structure matches PaddlePaddle's state_dict key names for
    direct weight conversion.
    """

    def __init__(self):
        super().__init__()

        # ---- resnet_head: 2 conv layers with stride=2 each ----
        # Paddle keys: resnet_head.0.weight, resnet_head.1._mean, resnet_head.3.weight, resnet_head.4._mean
        # (indices 2 and 5 are ReLU in forward, not stored)
        self.resnet_head = nn.Sequential()
        self.resnet_head.add_module('0', nn.Conv2d(3, 32, 5, stride=2, padding=2, bias=False))
        self.resnet_head.add_module('1', nn.BatchNorm2d(32))
        # index 2: ReLU (functional)
        self.resnet_head.add_module('3', nn.Conv2d(32, 32, 5, stride=2, padding=2, bias=False))
        self.resnet_head.add_module('4', nn.BatchNorm2d(32))
        # index 5: ReLU (functional)

        # ---- resnet_down: 3 stages of dilated residual blocks ----
        # Paddle stores under "resnet_down.layer{1,2,3}.{idx}.{bn,conv,downsample}"
        self.resnet_down = nn.ModuleDict()

        # Layer 1: 32→32, stride=1, dilations [1,2,4]
        self.resnet_down['layer1'] = nn.Sequential()
        self.resnet_down['layer1'].add_module('0', self._make_block(32, 32, stride=1, dilation=1, has_downsample=False, use_seq_conv=False))
        self.resnet_down['layer1'].add_module('1', self._make_block(32, 32, stride=1, dilation=2, has_downsample=False, use_seq_conv=True))
        self.resnet_down['layer1'].add_module('2', self._make_block(32, 32, stride=1, dilation=4, has_downsample=False, use_seq_conv=True))

        # Layer 2: 32→64, stride=2, dilations [1,1,2,4]
        self.resnet_down['layer2'] = nn.Sequential()
        self.resnet_down['layer2'].add_module('0', self._make_block(32, 64, stride=2, dilation=1, has_downsample=True, use_seq_conv=False))
        self.resnet_down['layer2'].add_module('1', self._make_block(64, 64, stride=1, dilation=1, has_downsample=False, use_seq_conv=True))
        self.resnet_down['layer2'].add_module('2', self._make_block(64, 64, stride=1, dilation=2, has_downsample=False, use_seq_conv=True))
        self.resnet_down['layer2'].add_module('3', self._make_block(64, 64, stride=1, dilation=4, has_downsample=False, use_seq_conv=True))

        # Layer 3: 64→128, stride=2, dilations [1,1,2,4,8,16]
        self.resnet_down['layer3'] = nn.Sequential()
        self.resnet_down['layer3'].add_module('0', self._make_block(64, 128, stride=2, dilation=1, has_downsample=True, use_seq_conv=False))
        self.resnet_down['layer3'].add_module('1', self._make_block(128, 128, stride=1, dilation=1, has_downsample=False, use_seq_conv=True))
        self.resnet_down['layer3'].add_module('2', self._make_block(128, 128, stride=1, dilation=2, has_downsample=False, use_seq_conv=True))
        self.resnet_down['layer3'].add_module('3', self._make_block(128, 128, stride=1, dilation=4, has_downsample=False, use_seq_conv=True))
        self.resnet_down['layer3'].add_module('4', self._make_block(128, 128, stride=1, dilation=8, has_downsample=False, use_seq_conv=True))
        self.resnet_down['layer3'].add_module('5', self._make_block(128, 128, stride=1, dilation=16, has_downsample=False, use_seq_conv=True))

        # ---- Spatial Pyramid (6 parallel dilated conv 3x3 + BN) ----
        # Paddle structure: bridges 1-3 have 1 sub-block each,
        # bridges 4-6 have 3 sub-blocks each.
        # Each sub-block is Sequential(Conv, BN) producing keys like:
        #   bridge_1.0.0.weight, bridge_1.0.1.running_mean
        #   bridge_4.0.0.weight, bridge_4.0.1.running_mean,
        #   bridge_4.1.0.weight, bridge_4.1.1.running_mean,
        #   bridge_4.2.0.weight, bridge_4.2.1.running_mean

        def make_bridge_block(in_ch, out_ch, dilation, num_sub_blocks):
            """Create a bridge module with `num_sub_blocks` Conv+BN pairs."""
            seq = nn.Sequential()
            for idx in range(num_sub_blocks):
                block = nn.Sequential()
                block.add_module('0', nn.Conv2d(in_ch, out_ch, 3, stride=1,
                                                 padding=dilation, dilation=dilation,
                                                 bias=False))
                block.add_module('1', nn.BatchNorm2d(out_ch))
                seq.add_module(str(idx), block)
            return seq

        self.bridge_1 = make_bridge_block(128, 128, 1, 1)
        self.bridge_2 = make_bridge_block(128, 128, 2, 1)
        self.bridge_3 = make_bridge_block(128, 128, 4, 1)
        self.bridge_4 = make_bridge_block(128, 128, 8, 3)
        self.bridge_5 = make_bridge_block(128, 128, 16, 3)
        self.bridge_6 = make_bridge_block(128, 128, 32, 3)

        # ---- Bridge concat: 768 (128*6) → 128 ----
        # Paddle keys: bridge_concat.0.weight, bridge_concat.1._mean, ...
        self.bridge_concat = nn.Sequential(
            nn.Conv2d(768, 128, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
        )

        # ---- Output Heads ----
        # Paddle keys: out_point_positions2D.0.*, .1.*, .2._weight, .3.*
        self.out_point_positions2D = nn.Sequential()
        self.out_point_positions2D.add_module('0', nn.Conv2d(128, 32, 5, stride=1, padding=2, bias=False))
        self.out_point_positions2D.add_module('1', nn.BatchNorm2d(32))
        self.out_point_positions2D.add_module('2', LearnableScale())
        self.out_point_positions2D.add_module('3', nn.Conv2d(32, 2, 5, stride=1, padding=2, bias=True))

        # Paddle keys: out_point_positions3D.0.*, .1.*, .2._weight, .3.*
        self.out_point_positions3D = nn.Sequential()
        self.out_point_positions3D.add_module('0', nn.Conv2d(128, 32, 5, stride=1, padding=2, bias=False))
        self.out_point_positions3D.add_module('1', nn.BatchNorm2d(32))
        self.out_point_positions3D.add_module('2', LearnableScale())
        self.out_point_positions3D.add_module('3', nn.Conv2d(32, 3, 5, stride=1, padding=2, bias=True))

    def _make_block(self, in_ch, out_ch, stride=1, dilation=1, has_downsample=False, use_seq_conv=False):
        """
        Create a dilated residual block with Paddle-compatible weight keys.

        Key patterns (matching Paddle):
        - use_seq_conv=False: bn1, conv1, bn2, conv2 (first block per stage)
        - use_seq_conv=True:  bn1, conv1.0, bn2, conv2.0 (subsequent blocks)
        - has_downsample=True: adds downsample.0 (Conv2d), downsample.1 (BN)
        """
        block = nn.ModuleDict()

        block['bn1'] = nn.BatchNorm2d(in_ch)
        # conv1: stride applied here for downsampling blocks
        if use_seq_conv:
            block['conv1'] = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 5, stride=stride, padding=2, dilation=1, bias=True)
            )
        else:
            block['conv1'] = nn.Conv2d(in_ch, out_ch, 5, stride=stride, padding=2, dilation=1, bias=True)

        block['bn2'] = nn.BatchNorm2d(out_ch)
        # conv2: dilation applied here
        pad = 2 * dilation
        if use_seq_conv:
            block['conv2'] = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 5, stride=1, padding=pad, dilation=dilation, bias=True)
            )
        else:
            block['conv2'] = nn.Conv2d(out_ch, out_ch, 5, stride=1, padding=pad, dilation=dilation, bias=True)

        if has_downsample:
            block['downsample'] = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 5, stride=stride, padding=2, bias=True),
                nn.BatchNorm2d(out_ch),
            )

        return block

    def _forward_block(self, block, x):
        """Forward through a dilated residual block (ModuleDict)."""
        identity = x

        out = block['bn1'](x)
        out = F.relu(out, inplace=True)
        out = block['conv1'](out)

        out = block['bn2'](out)
        out = F.relu(out, inplace=True)
        out = block['conv2'](out)

        if 'downsample' in block:
            identity = block['downsample'](x)
        else:
            if block['conv1'](x) is not None:
                pass  # no downsample needed

        out = out + identity
        return out

    def forward(self, x):
        # ---- Encoder ----
        # resnet_head: Conv+BN (0,1), ReLU, Conv+BN (3,4), ReLU
        x = self.resnet_head._modules['0'](x)
        x = self.resnet_head._modules['1'](x)
        x = F.relu(x, inplace=True)
        x = self.resnet_head._modules['3'](x)
        x = self.resnet_head._modules['4'](x)
        x = F.relu(x, inplace=True)

        # layer1
        for block in self.resnet_down['layer1']:
            x = self._forward_block(block, x)

        # layer2
        for block in self.resnet_down['layer2']:
            x = self._forward_block(block, x)

        # layer3
        for block in self.resnet_down['layer3']:
            x = self._forward_block(block, x)

        # ---- Spatial Pyramid ----
        b1 = self.bridge_1(x)
        b2 = self.bridge_2(x)
        b3 = self.bridge_3(x)
        b4 = self.bridge_4(x)
        b5 = self.bridge_5(x)
        b6 = self.bridge_6(x)
        x = torch.cat([b1, b2, b3, b4, b5, b6], dim=1)
        x = self.bridge_concat(x)  # Conv1x1+BN
        x = F.relu(x, inplace=True)

        # ---- Output Heads ----
        # out_point_positions2D: Conv+BN (0,1), ReLU, Scale (2), Conv (3)
        g2d = self.out_point_positions2D._modules['0'](x)
        g2d = self.out_point_positions2D._modules['1'](g2d)
        g2d = F.relu(g2d, inplace=True)
        g2d = self.out_point_positions2D._modules['2'](g2d)   # LearnableScale
        g2d = self.out_point_positions2D._modules['3'](g2d)   # Final Conv: 32→2

        # out_point_positions3D: Conv+BN (0,1), ReLU, Scale (2), Conv (3)
        g3d = self.out_point_positions3D._modules['0'](x)
        g3d = self.out_point_positions3D._modules['1'](g3d)
        g3d = F.relu(g3d, inplace=True)
        g3d = self.out_point_positions3D._modules['2'](g3d)   # LearnableScale
        g3d = self.out_point_positions3D._modules['3'](g3d)   # Final Conv: 32→3

        return {"unwarp_grid": g2d, "mesh_3d": g3d}

    def unwarp(self, image, output_size=None):
        """
        Unwarp a document image using the predicted 2D grid.

        Args:
            image: Input tensor [B, 3, H, W]
            output_size: (H_out, W_out) tuple, or None to use input size

        Returns:
            unwarped_image: [B, 3, H_out, W_out]
            unwarp_grid_norm: [B, H_out, W_out, 2] in [-1, 1] normalized coords
        """
        if output_size is None:
            output_size = (image.shape[2], image.shape[3])

        result = self.forward(image)
        grid = result["unwarp_grid"]  # [B, 2, grid_h, grid_w]

        # Bilinearly interpolate grid to full resolution
        grid_full = F.interpolate(grid, size=output_size,
                                  mode='bilinear', align_corners=True)
        # Convert to grid_sample format: [B, H, W, 2], normalized to [-1, 1]
        B, _, H, W = grid_full.shape
        grid_norm = grid_full.permute(0, 2, 3, 1)
        grid_norm[..., 0] = grid_norm[..., 0] / (image.shape[3] - 1) * 2.0 - 1.0
        grid_norm[..., 1] = grid_norm[..., 1] / (image.shape[2] - 1) * 2.0 - 1.0

        unwarped = F.grid_sample(image, grid_norm, mode='bilinear',
                                 padding_mode='zeros', align_corners=True)
        return unwarped, grid_norm
