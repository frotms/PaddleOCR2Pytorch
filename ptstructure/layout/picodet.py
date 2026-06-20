"""
PP-DocLayout: PyTorch PicoDet Layout Detection Model.
Architecture: LCNet + LCPAN + PicoHeadV2.

Supports multiple variants:
  - S:  backbone [16,24,48,96,192,384],  neck/head 96ch
  - M:  backbone [32,64,128,256,512,1024], neck/head 160ch

Key names match the Paddle state dict for direct weight loading via converter.

Usage:
    # S variant
    model = PPDocLayout(variant='S')
    model.load_state_dict(torch.load('ptocr_ppdoclayout_s.pth'))

    # M variant
    model = PPDocLayout(variant='M')
    model.load_state_dict(torch.load('ptocr_ppdoclayout_m.pth'))

    detections = model.detect(img)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


# ---------------------------------------------------------------------------
#  Variant configurations
# ---------------------------------------------------------------------------

_VARIANTS = {
    # name: (c1, c2, c3, c4, c5, c6, neck_head_ch, num_convs)
    'S': (16, 24, 48, 96, 192, 384, 96, 2),
    'M': (32, 64, 128, 256, 512, 1024, 160, 4),
    'L': (64, 128, 256, 512, 1024, 2048, 256, 4),
}


# ---------------------------------------------------------------------------
#  Common building blocks
# ---------------------------------------------------------------------------

class ConvBN(nn.Module):
    """Conv + BN + optional activation. Used in backbone and neck.
    Paddle counterpart: LCNet.ConvBNLayer / CSPPAN.ConvBNLayer.

    Key names: .conv.weight, .bn.weight, .bn.bias, .bn.running_mean, .bn.running_var
    """
    def __init__(self, in_ch, out_ch, kernel, stride, groups=1, act='hard_swish'):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, kernel // 2,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, momentum=0.9)
        self.act = act
        self._act_fn = _get_act(act)

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self._act_fn is not None:
            x = self._act_fn(x)
        return x


class HeadConvNorm(nn.Module):
    """Conv + BN WITHOUT activation. Used in PicoFeat (head).
    Paddle counterpart: ConvNormLayer in ppdet.modeling.layers.

    Key names: .conv.weight, .norm.weight, .norm.bias, .norm.running_mean, .norm.running_var
    """
    def __init__(self, in_ch, out_ch, kernel, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, kernel // 2,
                              groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.norm(self.conv(x))


class InvertedResidual(nn.Module):
    """LCNet-style DepthwiseSeparable.
    Paddle counterpart: LCNet.DepthwiseSeparable.

    Key names: .dw_conv.{conv,bn}, .pw_conv.{conv,bn}
    Optional SE: .se.{conv1,conv2}

    NOTE: Paddle LCNet DepthwiseSeparable does NOT have a skip connection
    (unlike MobileNetV3). This implementation matches the Paddle behavior exactly.
    """
    def __init__(self, in_ch, out_ch, stride, kernel, act='hard_swish', se_mid=None):
        super().__init__()
        self.dw_conv = ConvBN(in_ch, in_ch, kernel, stride, groups=in_ch, act=act)
        self.se = None
        if se_mid is not None:
            self.se = nn.Module()
            self.se.conv1 = nn.Conv2d(in_ch, se_mid, 1)
            self.se.conv2 = nn.Conv2d(se_mid, in_ch, 1)
        self.pw_conv = ConvBN(in_ch, out_ch, 1, 1, act=act)

    def forward(self, x):
        out = self.dw_conv(x)          # conv + bn + act
        if self.se is not None:
            w = F.adaptive_avg_pool2d(out, 1)
            w = F.relu(self.se.conv1(w))
            w = F.hardsigmoid(self.se.conv2(w))
            out = out * w
        out = self.pw_conv(out)        # conv + bn + act
        return out


class DPModule(nn.Module):
    """Depth-wise + Point-wise module with optional final activation.
    Paddle counterpart: CSPPAN.DPModule (ppdet.modeling.necks.csp_pan).

    Key names: .dwconv.weight, .bn1.*, .pwconv.weight, .bn2.*
    """
    def __init__(self, in_ch, out_ch, kernel, stride, act='hard_swish',
                 use_act_in_out=True):
        super().__init__()
        self.dwconv = nn.Conv2d(in_ch, out_ch, kernel, stride, kernel // 2,
                                groups=out_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch, momentum=0.9)
        self.pwconv = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch, momentum=0.9)
        self.act = act
        self.use_act_in_out = use_act_in_out
        self._act_fn = _get_act(act)

    def forward(self, x):
        x = self.bn1(self.dwconv(x))
        if self._act_fn is not None:
            x = self._act_fn(x)
        x = self.bn2(self.pwconv(x))
        if self.use_act_in_out and self._act_fn is not None:
            x = self._act_fn(x)
        return x


class CSPLayer(nn.Module):
    """Simplified CSP block – two InvertedResidual blocks in series.
    This matches the PicoDet CSPLayer (PaddleDetection csp_pan.py) where
    the blocks are depthwise-separable DarknetBottleneck-style blocks,
    serialized as two InvertedResidual modules.

    Paddle counterpart: CSPLayer with num_csp_blocks=1.

    Key names: .0.dw_conv, .0.pw_conv, .1.dw_conv, .1.pw_conv
    """
    def __init__(self, in_ch, out_ch, mid_ch, kernel=5, act='hard_swish'):
        super().__init__()
        self.add_module('0', InvertedResidual(in_ch, mid_ch, 1, kernel, act=act))
        self.add_module('1', InvertedResidual(mid_ch, out_ch, 1, kernel, act=act))

    def forward(self, x):
        return self._modules['1'](self._modules['0'](x))


# ---------------------------------------------------------------------------
#  Head sub-modules
# ---------------------------------------------------------------------------

class PicoSE(nn.Module):
    """SE module for PicoFeat.
    Paddle counterpart: PicoSE in pico_head.py.

    Key names: .fc.{weight,bias}, .conv.conv.weight, .conv.norm.*
    Forward: avg_pool → fc → sigmoid → multiply → conv+bn → (act applied externally)
    """
    def __init__(self, feat_channels):
        super().__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
        self.conv = nn.Module()
        self.conv.conv = nn.Conv2d(feat_channels, feat_channels, 1, bias=False)
        self.conv.norm = nn.BatchNorm2d(feat_channels)

    def forward(self, feat, avg_feat):
        weight = torch.sigmoid(self.fc(avg_feat))
        out = self.conv.norm(self.conv.conv(feat * weight))
        return out


class AlignDPModule(nn.Module):
    """DPModule for cls_align head (use_act_in_out=False, out_ch=1).
    Paddle counterpart: DPModule(feat_in_chan, 1, 5, act, use_act_in_out=False).

    Key names: .dwconv.weight, .bn1.*, .pwconv.weight, .bn2.*

    NOTE: Paddle's DPModule always does groups=out_channel for dwconv.
    Since out_channel=1 for cls_align, groups=1 (no depthwise), which matches
    the Paddle weights: dwconv.weight shape [1, nc, 5, 5].
    """
    def __init__(self, in_ch, act='hard_swish'):
        super().__init__()
        self.dwconv = nn.Conv2d(in_ch, 1, 5, 1, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(1)
        self.pwconv = nn.Conv2d(1, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(1)
        self.act = act
        self._act_fn = _get_act(act)

    def forward(self, x):
        x = self.bn1(self.dwconv(x))
        if self._act_fn is not None:
            x = self._act_fn(x)
        x = self.bn2(self.pwconv(x))
        return x


def _get_act(act_name):
    if act_name is None:
        return None
    if act_name == 'relu':
        return nn.ReLU(inplace=True)
    if act_name in ('hard_swish', 'hardswish'):
        return nn.Hardswish(inplace=True)
    if act_name == 'relu6':
        return nn.ReLU6(inplace=True)
    if act_name == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    raise ValueError(f'Unknown activation: {act_name}')


# ---------------------------------------------------------------------------
#  Main Model
# ---------------------------------------------------------------------------

class PPDocLayout(nn.Module):
    """PP-DocLayout layout detection model (PicoDet with PicoHeadV2).

    Supports variants: 'S', 'M', 'L', or explicit channel config.

    Architecture: LCNet backbone + LCPAN neck + PicoHeadV2 head.

    Usage:
        model = PPDocLayout(variant='M')
        model.load_state_dict(torch.load('ptocr_ppdoclayout_m.pth'))
        detections = model.detect(img)
    """

    def __init__(self, variant='M', num_classes=23, neck_head_ch=None,
                 backbone_channels=None, num_convs=None):
        """
        Args:
            variant: 'S', 'M', or 'L' (overridden by explicit channel params).
            num_classes: number of layout classes (default 23).
            neck_head_ch: explicit neck/head channel count.
            backbone_channels: explicit (c1,c2,c3,c4,c5,c6) tuple.
            num_convs: number of dw+pw conv pairs in PicoFeat (default: variant-specific).
        """
        super().__init__()
        if backbone_channels is not None and neck_head_ch is not None:
            c1, c2, c3, c4, c5, c6 = backbone_channels
            nc = neck_head_ch
        elif variant in _VARIANTS:
            c1, c2, c3, c4, c5, c6, nc, nc_nconv = _VARIANTS[variant]
            if num_convs is None:
                num_convs = nc_nconv
        else:
            raise ValueError(f"Unknown variant: {variant}. Choose from {list(_VARIANTS.keys())} "
                             "or provide backbone_channels and neck_head_ch explicitly.")

        self.num_classes = num_classes
        self.reg_max = 7
        self.cell_offset = 0.5
        self.variant = variant
        self.neck_head_ch = nc
        act = 'hard_swish'

        # ---- Backbone ----
        self.backbone = nn.Module()
        self.backbone.conv1 = ConvBN(3, c1, 3, 2, act=act)
        self.backbone.blocks2 = nn.Sequential(InvertedResidual(c1, c2, 1, 3, act=act))
        self.backbone.blocks3 = nn.Sequential(
            InvertedResidual(c2, c3, 2, 3, act=act),
            InvertedResidual(c3, c3, 1, 3, act=act))
        self.backbone.blocks4 = nn.Sequential(
            InvertedResidual(c3, c4, 2, 3, act=act),
            InvertedResidual(c4, c4, 1, 3, act=act))
        self.backbone.blocks5 = nn.Sequential(
            InvertedResidual(c4, c5, 2, 3, act=act),
            *[InvertedResidual(c5, c5, 1, 5, act=act) for _ in range(5)])
        self.backbone.blocks6 = nn.Sequential(
            InvertedResidual(c5, c6, 2, 5, act=act, se_mid=c5 // 4),
            InvertedResidual(c6, c6, 1, 5, act=act, se_mid=c6 // 4))

        # ---- Neck (LCPAN) ----
        self.neck = nn.Module()
        nk_ct = nn.Module()
        nk_ct.convs = nn.Module()
        for i, ch in enumerate([c4, c5, c6]):
            nk_ct.convs.add_module(str(i), ConvBN(ch, nc, 1, 1, act=act))
        self.neck.conv_t = nk_ct
        self.neck.first_top_conv = DPModule(nc, nc, 5, 2, act=act, use_act_in_out=True)
        self.neck.second_top_conv = DPModule(nc, nc, 5, 2, act=act, use_act_in_out=True)
        nk_td = nn.Module()
        nk_td.add_module('0', CSPLayer(nc * 2, nc, nc * 2, 5, act=act))
        nk_td.add_module('1', CSPLayer(nc * 2, nc, nc * 2, 5, act=act))
        self.neck.top_down_blocks = nk_td
        nk_bu = nn.Module()
        nk_bu.add_module('0', CSPLayer(nc * 2, nc, nc * 2, 5, act=act))
        nk_bu.add_module('1', CSPLayer(nc * 2, nc, nc * 2, 5, act=act))
        self.neck.bottom_up_blocks = nk_bu
        nk_ds = nn.Module()
        for i in range(2):
            nk_ds.add_module(str(i), DPModule(nc, nc, 5, 2, act=act, use_act_in_out=True))
        self.neck.downsamples = nk_ds

        # ---- Head (PicoHeadV2) ----
        self.head = nn.Module()
        self.head.conv_feat = nn.Module()
        for stage_idx in range(4):
            dw_container = nn.Module()
            pw_container = nn.Module()
            for i in range(num_convs):
                dw_container.add_module(str(i), HeadConvNorm(nc, nc, 5, groups=nc))
                pw_container.add_module(str(i), HeadConvNorm(nc, nc, 1))
            self.head.conv_feat.add_module('cls_conv_dw{}'.format(stage_idx), dw_container)
            self.head.conv_feat.add_module('cls_conv_pw{}'.format(stage_idx), pw_container)
        hd_se = nn.Module()
        for i in range(4):
            hd_se.add_module(str(i), PicoSE(nc))
        self.head.conv_feat.se = hd_se
        self.head.cls_align = nn.Module()
        for i in range(4):
            self.head.cls_align.add_module(str(i), AlignDPModule(nc, act=act))
        mod_cls = nn.Module()
        for i in range(4):
            mod_cls.add_module(str(i), nn.Conv2d(nc, num_classes, 1))
        self.head.head_cls_list = mod_cls

        mod_reg = nn.Module()
        for i in range(4):
            mod_reg.add_module(str(i), nn.Conv2d(nc, 4 * (self.reg_max + 1), 1))
        self.head.head_reg_list = mod_reg

        # Distribution project buffer
        self.register_buffer('proj', torch.arange(self.reg_max + 1, dtype=torch.float32))

        # Dummy scale_reg modules (present in Paddle weights, unused in forward)
        for prefix in ['p3_feat', 'p4_feat', 'p5_feat', 'p6_feat']:
            self.head.add_module(prefix, _ScaleRegModule())

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, x):
        bb = self.backbone
        nk = self.neck
        hd = self.head
        B, _, H_in, W_in = x.shape

        # ---- Backbone ----
        x = bb.conv1(x)                # s2  (hard_swish inside ConvBN)
        x = bb.blocks2(x)
        x = bb.blocks3(x)
        x = bb.blocks4(x); c4 = x
        x = bb.blocks5(x); c5 = x
        x = bb.blocks6(x); c6 = x

        # ---- Neck (LCPAN) ----
        # Channel tuning: all -> nc
        l4 = nk.conv_t.convs._modules['0'](c4)   # s16
        l5 = nk.conv_t.convs._modules['1'](c5)   # s32
        l6 = nk.conv_t.convs._modules['2'](c6)   # s64
        inputs = [l4, l5, l6]

        # Top-down
        inner_outs = [inputs[-1]]
        for idx in range(2, 0, -1):   # idx = 2, 1
            feat_high = inner_outs[0]
            feat_low = inputs[idx - 1]
            upsample_feat = F.interpolate(feat_high, size=feat_low.shape[2:], mode='nearest')
            block = nk.top_down_blocks._modules[str(2 - idx)]
            inner_out = block(torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # Bottom-up
        outs = [inner_outs[0]]
        for idx in range(2):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            ds = nk.downsamples._modules[str(idx)]
            bu = nk.bottom_up_blocks._modules[str(idx)]
            downsample_feat = ds(feat_low)
            out = bu(torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)

        # P7: generated top feature by downsampling top input and top output
        top_feat = nk.first_top_conv(inputs[-1]) + nk.second_top_conv(outs[-1])
        outs.append(top_feat)          # outs = [P4(s8), P5(s16), P6(s32), P7(s64)]
        feats = outs                   # [s8, s16, s32, s64] — all 4 neck outputs used directly
        strides = [8, 16, 32, 64]

        # ---- Head (PicoHeadV2) ----
        eps = 1e-9
        all_cls, all_boxes = [], []

        for i, (feat, stride) in enumerate(zip(feats, strides)):
            _, _, H, W = feat.shape

            # ---- PicoFeat ----
            # Interleaved dw/pw: dw0->act->pw0->act->dw1->act->pw1->act->...
            dw_seq = hd.conv_feat._modules['cls_conv_dw{}'.format(i)]
            pw_seq = hd.conv_feat._modules['cls_conv_pw{}'.format(i)]
            xf = feat
            for j in range(len(dw_seq._modules)):
                dw_layer = dw_seq._modules[str(j)]
                pw_layer = pw_seq._modules[str(j)]
                xf = F.hardswish(dw_layer(xf))
                xf = F.hardswish(pw_layer(xf))

            conv_cls_feat = xf  # save before SE for alignment

            # PicoSE: avg_pool -> fc -> sigmoid -> multiply -> conv+bn -> hardswish
            se = hd.conv_feat.se._modules[str(i)]
            avg_feat = F.adaptive_avg_pool2d(conv_cls_feat, 1)
            se_feat = F.hardswish(se(conv_cls_feat, avg_feat))

            # Classification logit (from SE feature)
            cls_logit = hd.head_cls_list._modules[str(i)](se_feat)

            # Classification alignment (from non-SE feature)
            cls_prob = torch.sigmoid(hd.cls_align._modules[str(i)](conv_cls_feat))

            # Combined score: sqrt(sigmoid(logit) * align_prob + eps)
            cls_score = torch.sqrt(torch.sigmoid(cls_logit) * cls_prob + eps)

            # Regression prediction (from SE feature)
            reg_pred = hd.head_reg_list._modules[str(i)](se_feat)

            # DFL decode: softmax + weighted sum
            # reg_pred: [B, 32, H, W] -> [B, H, W, 4, 8]
            reg_pred = reg_pred.permute(0, 2, 3, 1).reshape(B, H, W, 4, self.reg_max + 1)
            ltrb = (F.softmax(reg_pred, dim=-1) * self.proj.to(reg_pred.device)).sum(dim=-1)
            # ltrb: [B, H, W, 4] -> [l, t, r, b] in stride units

            # Grid with cell_offset (0.5 for PP-DocLayout)
            ys = torch.arange(H, dtype=torch.float32, device=x.device) + self.cell_offset
            xs = torch.arange(W, dtype=torch.float32, device=x.device) + self.cell_offset
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
            # anchor_point: [H, W, 2] -> [x, y]
            anchor = torch.stack([xx, yy], dim=-1)  # [H, W, 2]

            # distance2bbox: [x - l, y - t, x + r, y + b]
            lt = ltrb[..., :2]   # l, t
            rb = ltrb[..., 2:]   # r, b
            boxes = torch.cat([anchor - lt, anchor + rb], dim=-1) * stride
            # boxes: [B, H, W, 4]

            all_cls.append(cls_score.reshape(B, self.num_classes, H * W))
            all_boxes.append(boxes.reshape(B, H * W, 4))

        cls_all = torch.cat(all_cls, dim=2)     # [B, num_classes, total_anchors]
        boxes_all = torch.cat(all_boxes, dim=1)  # [B, total_anchors, 4]
        return cls_all, boxes_all, strides

    # ------------------------------------------------------------------
    #  Detection pipeline
    # ------------------------------------------------------------------
    def detect(self, img, score_thresh=0.3, nms_thresh=0.5, max_dets=100,
               input_size=640):
        """Run detection on BGR image.

        Args:
            img: BGR image (H, W, 3), uint8 or float32.
            score_thresh: minimum score threshold (default 0.3).
            nms_thresh: IoU threshold for NMS (default 0.5).
            max_dets: maximum number of detections (keep_top_k: 100).
            input_size: resize input to (input_size, input_size).

        Returns:
            List of dicts: [{label, score, bbox:[x1,y1,x2,y2]}, ...]
        """
        oh, ow = img.shape[:2]
        # PP-DocLayout preprocessing: direct resize to input_size x input_size
        resized = cv2.resize(img, (input_size, input_size))
        t = (resized.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        x = torch.from_numpy(t.transpose(2, 0, 1)).float().unsqueeze(0)

        with torch.no_grad():
            cls_all, boxes_all, _ = self.forward(x)

        # Take batch 0
        cls_all = cls_all[0]     # [num_classes, N]
        boxes_all = boxes_all[0]  # [N, 4]

        scores, labels = cls_all.max(dim=0)
        mask = scores > score_thresh
        if mask.sum() == 0:
            return []

        boxes = boxes_all[mask]
        scores = scores[mask]
        labels = labels[mask]
        # Scale boxes from input_size back to original image size
        sx, sy = ow / float(input_size), oh / float(input_size)
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
        # Clamp to image boundaries
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, ow)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, oh)

        # Per-class NMS
        kept_indices = []
        for cid in labels.unique():
            m = labels == cid
            cb = boxes[m]
            cs = scores[m]
            order = cs.argsort(descending=True)
            while len(order) > 0:
                i = order[0]
                kept_indices.append((torch.where(m)[0][i].item(), cs[i].item()))
                if len(order) == 1:
                    break
                xx1 = torch.max(cb[i, 0], cb[order[1:], 0])
                yy1 = torch.max(cb[i, 1], cb[order[1:], 1])
                xx2 = torch.min(cb[i, 2], cb[order[1:], 2])
                yy2 = torch.min(cb[i, 3], cb[order[1:], 3])
                inter = torch.clamp(xx2 - xx1, 0) * torch.clamp(yy2 - yy1, 0)
                ai = (cb[i, 2] - cb[i, 0]) * (cb[i, 3] - cb[i, 1])
                ao = (cb[order[1:], 2] - cb[order[1:], 0]) * (cb[order[1:], 3] - cb[order[1:], 1])
                order = order[1:][inter / (ai + ao - inter + 1e-6) < nms_thresh]

        kept_indices.sort(key=lambda x: x[1], reverse=True)
        keep = torch.tensor([idx for idx, _ in kept_indices[:max_dets]])

        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        # Label order from PaddleX PP-DocLayout-S inference.yml (training label taxonomy)
        label_names = [
            'paragraph_title', 'image', 'text', 'number', 'abstract', 'content',
            'figure_title', 'formula', 'table', 'table_title', 'reference',
            'doc_title', 'footnote', 'header', 'algorithm', 'footer', 'seal',
            'chart_title', 'chart', 'formula_number', 'header_image',
            'footer_image', 'aside_text']
        return [{'label': label_names[labels[i]], 'score': float(scores[i]),
                 'bbox': [int(boxes[i, 0]), int(boxes[i, 1]),
                          int(boxes[i, 2]), int(boxes[i, 3])]}
                for i in range(len(boxes))]


class _ScaleRegModule(nn.Module):
    """Dummy module to hold the scale_reg parameter from Paddle weights.
    Not used in forward pass but needed for state dict compatibility.
    """
    def __init__(self):
        super().__init__()
        self.register_parameter('scale_reg', nn.Parameter(torch.ones(1)))

    def forward(self, x):
        return x
