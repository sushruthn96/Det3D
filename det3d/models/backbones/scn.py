import time

import numpy as np
import spconv
import torch
from det3d.models.utils import Empty, change_default_args
from det3d.torchie.cnn import constant_init, kaiming_init
from det3d.torchie.trainer import load_checkpoint
from spconv import SparseConv3d, SubMConv3d
from torch import nn
from torch.nn import BatchNorm1d
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.ops import pts_in_boxes3d
from mmdet.ops.pointnet2 import pointnet2_utils
from .. import builder
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer, change_default_args, Sequential

from mmdet.core import tensor2points


def conv3x3(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """3x3 convolution with padding"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )


def conv1x1(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """1x1 convolution"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        downsample=None,
        indice_key=None,
    ):
        super(SparseBasicBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv1 = conv3x3(inplanes, planes, stride, indice_key=indice_key, bias=bias)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, indice_key=indice_key, bias=bias)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity.features
        out.features = self.relu(out.features)

        return out


@BACKBONES.register_module
class SpMiddleFHD(nn.Module):
    def __init__(self,
                 output_shape, ds_factor=8,
                 num_input_features=4,
                 num_hidden_features=128,
                 ):

        super(SpMiddleFHD, self).__init__()

#         print(output_shape)
#         self.sparse_shape = output_shape
        self.sparse_shape = None

        self.backbone1 = VxNet(num_input_features)
#         self.fcn = BEVNet(in_features=num_hidden_features, num_filters=256)
        
#             def __init__(
#         self, num_input_features=128, norm_cfg=None, name="SpMiddleResNetFHD", **kwargs
#     ):
#         super(SpMiddleResNetFHD, self).__init__()
#         self.name = name

#         self.dcn = None
#         self.zero_init_residual = False

#         if norm_cfg is None:
#             norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

    def _make_layer(self, conv2d, bachnorm2d, inplanes, planes, num_blocks, stride=1):
        block = Sequential(
            nn.ZeroPad2d(1),
            conv2d(inplanes, planes, 3, stride=stride),
            bachnorm2d(planes),
            nn.ReLU(),
        )
        for j in range(num_blocks):
            block.add(conv2d(planes, planes, 3, padding=1))
            block.add(bachnorm2d(planes))
            block.add(nn.ReLU())
        return block, planes

    def build_aux_target(self, nxyz, gt_boxes3d, enlarge=1.0):
        center_offsets = list()
        pts_labels = list()
        for i in range(len(gt_boxes3d)):
            boxes3d = gt_boxes3d[i].cpu()
            idx = torch.nonzero(nxyz[:, 0] == i).view(-1)
            new_xyz = nxyz[idx, 1:].cpu()

            boxes3d[:, 3:6] *= enlarge

            pts_in_flag, center_offset = pts_in_boxes3d(new_xyz, boxes3d)
            pts_label = pts_in_flag.max(0)[0].byte()

            # import mayavi.mlab as mlab
            # from mmdet.datasets.kitti_utils import draw_lidar, draw_gt_boxes3d
            # f = draw_lidar((new_xyz).numpy(), show=False)
            # pts = new_xyz[pts_label].numpy()
            # mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], color=(1, 1, 1), scale_factor=0.25, figure=f)
            # f = draw_gt_boxes3d(center_to_corner_box3d(boxes3d.numpy()), f, draw_text=False, show=True)

            pts_labels.append(pts_label)
            center_offsets.append(center_offset)

        center_offsets = torch.cat(center_offsets).cuda()
        pts_labels = torch.cat(pts_labels).cuda()

        return pts_labels, center_offsets

    def aux_loss(self, points, point_cls, point_reg, gt_bboxes):

        N = len(gt_bboxes)

        pts_labels, center_targets = self.build_aux_target(points, gt_bboxes)

        rpn_cls_target = pts_labels.float()
        pos = (pts_labels > 0).float()
        neg = (pts_labels == 0).float()

        pos_normalizer = pos.sum()
        pos_normalizer = torch.clamp(pos_normalizer, min=1.0)

        cls_weights = pos + neg
        cls_weights = cls_weights / pos_normalizer

        reg_weights = pos
        reg_weights = reg_weights / pos_normalizer

        aux_loss_cls = weighted_sigmoid_focal_loss(point_cls.view(-1), rpn_cls_target, weight=cls_weights, avg_factor=1.)
        aux_loss_cls /= N

        aux_loss_reg = weighted_smoothl1(point_reg, center_targets, beta=1 / 9., weight=reg_weights[..., None], avg_factor=1.)
        aux_loss_reg /= N

        return dict(
            aux_loss_cls = aux_loss_cls,
            aux_loss_reg = aux_loss_reg,
        )

    def forward(self, voxel_features, coors, batch_size, input_shape, is_test=False):

        points_mean = torch.zeros_like(voxel_features)
#         points_mean = torch.zeros((voxel_features.shape[0],4))
        points_mean[:, 0] = coors[:, 0]
        
        points_mean[:, 1:] = voxel_features[:, :4]
#         points_mean = points_mean.contiguous()
        
        is_test = False 
        coors = coors.int()
        self.sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]
#         print("coors, sparse, batch size",coors.shape, self.sparse_shape, batch_size)
        x = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
#         print("voxel features shape", voxel_features.shape)
        x, point_misc = self.backbone1(x, points_mean, is_test)
#         print("voxel features shape", voxel_features.shape)
  
        x = x.dense()
#         print("X shape beofre reshape!",x.shape)
        N, C, D, H, W = x.shape
        x = x.view(N, C * D, H, W)
#         print("X shape!",x.shape)
#         x = self.fcn(x)

        if is_test:
            return x

        return x, point_misc


def single_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 1, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
    )

def double_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
    )

def triple_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
    )

def stride_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
            spconv.SparseConv3d(in_channels, out_channels, 3, 2, padding=1, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()
    )

# def tensor2points(tensor, offset=(-50.4, -50.4, -5.0), voxel_size=(.05, .05, .1)):
#     indices = tensor.indices.float()
#     offset = torch.Tensor(offset).to(indices.device)
#     voxel_size = torch.Tensor(voxel_size).to(indices.device)
#     indices[:, 1:] = indices[:, [3, 2, 1]] * voxel_size + offset + .5 * voxel_size
#     return tensor.features, indices

def nearest_neighbor_interpolate(unknown, known, known_feats):
    """
    :param pts: (n, 4) tensor of the bxyz positions of the unknown features
    :param ctr: (m, 4) tensor of the bxyz positions of the known features
    :param ctr_feats: (m, C) tensor of features to be propigated
    :return:
        new_features: (n, C) tensor of the features of the unknown features
    """
    dist, idx = pointnet2_utils.three_nn(unknown, known)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm
    interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)

    return interpolated_feats


class VxNet(nn.Module):

    def __init__(self, num_input_features):
        super(VxNet, self).__init__()

        self.conv0 = double_conv(num_input_features, 16, 'subm0')
        self.down0 = stride_conv(16, 32, 'down0')

        self.conv1 = double_conv(32, 32, 'subm1')
        self.down1 = stride_conv(32, 64, 'down1')

        self.conv2 = triple_conv(64, 64, 'subm2')
        self.down2 = stride_conv(64, 64, 'down2')
        # increase channels
        self.conv3 = triple_conv(64, 64, 'subm3')  # middle line

        self.extra_conv = spconv.SparseSequential(
            spconv.SparseConv3d(64, 64, (1, 1, 1), (1, 1, 1), bias=False),  # shape no change
            nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        self.point_fc = nn.Linear(160, 64, bias=False)
        self.point_cls = nn.Linear(64, 1, bias=False)
        self.point_reg = nn.Linear(64, 3, bias=False)


    def forward(self, x, points_mean, is_test=False):
        
#         print("hello")
        x = self.conv0(x)
        x = self.down0(x)  # sp
        x = self.conv1(x)  # 2x sub
#         print("vxnet")
        if not is_test:
            vx_feat, vx_nxyz = tensor2points(x, voxel_size=(.2, .2, .4))
            p1 = nearest_neighbor_interpolate(points_mean[:,:-1].contiguous(), vx_nxyz, vx_feat)
#             print("vxnet")

#         print("X DENSE SHAPE", x.dense().shape)  [:,:-1].contiguous()
        x = self.down1(x)
        x = self.conv2(x)

        if not is_test:
            vx_feat, vx_nxyz = tensor2points(x, voxel_size=(.4, .4, .8))
            p2 = nearest_neighbor_interpolate(points_mean[:,:-1].contiguous(), vx_nxyz, vx_feat)
#             print("vxnet")

        x = self.down2(x)
        x = self.conv3(x)

        if not is_test:
            vx_feat, vx_nxyz = tensor2points(x, voxel_size=(.8, .8, 1.6))
#             print("VX_NXYZ, POINTS_MEAN", vx_nxyz[0:2, :], points_mean[0:2, :])
            p3 = nearest_neighbor_interpolate(points_mean[:,:-1].contiguous(), vx_nxyz, vx_feat)
       
            
#         print("vxnet")
#         print("X DENSE SHAPE", x.dense().shape)

        out = self.extra_conv(x)

        if is_test:
            return out, None
#         print("vxnet")
        pointwise = self.point_fc(torch.cat([p1, p2, p3], dim=-1))
        point_cls = self.point_cls(pointwise)
        point_reg = self.point_reg(pointwise)
        return out, (points_mean, point_cls, point_reg)

class BEVNet(nn.Module):
    def __init__(self, in_features, num_filters=256):
        super(BEVNet, self).__init__()
        BatchNorm2d = change_default_args(
            eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
        Conv2d = change_default_args(bias=False)(nn.Conv2d)

        self.conv0 = Conv2d(in_features, num_filters, 3, padding=1)
        self.bn0 = BatchNorm2d(num_filters)

        self.conv1 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn1 = BatchNorm2d(num_filters)

        self.conv2 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn2 = BatchNorm2d(num_filters)

        self.conv3 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn3 = BatchNorm2d(num_filters)

        self.conv4 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn4 = BatchNorm2d(num_filters)

        self.conv5 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn5 = BatchNorm2d(num_filters)

        self.conv6 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn6 = BatchNorm2d(num_filters)

        self.conv7 = Conv2d(num_filters, num_filters, 1)
        self.bn7 = BatchNorm2d(num_filters)

    def forward(self, x):
        x = self.conv0(x)
        x = F.relu(self.bn0(x), inplace=True)
        x = self.conv1(x)
        x = F.relu(self.bn1(x), inplace=True)
        x = self.conv2(x)
        x = F.relu(self.bn2(x), inplace=True)
        x = self.conv3(x)
        x = F.relu(self.bn3(x), inplace=True)
        x = self.conv4(x)
        x = F.relu(self.bn4(x), inplace=True)
        x = self.conv5(x)
        x = F.relu(self.bn5(x), inplace=True)
        x = self.conv6(x)
        x = F.relu(self.bn6(x), inplace=True)
        conv6 = x.clone()
        x = self.conv7(x)
        x = F.relu(self.bn7(x), inplace=True)
        return x, conv6




@BACKBONES.register_module
class SpMiddleFHDNobn(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="SpMiddleFHD", **kwargs
    ):
        super(SpMiddleFHDNobn, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, bias=True, indice_key="subm0"),
            # build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(),
            SubMConv3d(16, 16, 3, bias=True, indice_key="subm0"),
            # build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(),
            SparseConv3d(
                16, 32, 3, 2, padding=1, bias=True
            ),  # [1600, 1200, 41] -> [800, 600, 21]
            # build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1", bias=True),
            # build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1", bias=True),
            # build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SparseConv3d(
                32, 64, 3, 2, padding=1, bias=True
            ),  # [800, 600, 21] -> [400, 300, 11]
            # build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2", bias=True),
            # build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2", bias=True),
            # build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2", bias=True),
            # build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SparseConv3d(
                64, 64, 3, 2, padding=[0, 1, 1], bias=True
            ),  # [400, 300, 11] -> [200, 150, 5]
            # build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3", bias=True),
            # build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3", bias=True),
            # build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3", bias=True),
            # build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SparseConv3d(
                64, 64, (3, 1, 1), (2, 1, 1), bias=True
            ),  # [200, 150, 5] -> [200, 150, 2]
            # build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
        )

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(m, "conv2_offset"):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, voxel_features, coors, batch_size, input_shape):

        # input: # [41, 1600, 1408]
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]
        coors = coors.int()

        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        return ret


@BACKBONES.register_module
class SpMiddleResNetFHD(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="SpMiddleResNetFHD", **kwargs
    ):
        super(SpMiddleResNetFHD, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        # input: # [1600, 1200, 41]
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, bias=False, indice_key="res0"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(),
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
            SparseConv3d(
                16, 32, 3, 2, padding=1, bias=False
            ),  # [1600, 1200, 41] -> [800, 600, 21]
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            SparseConv3d(
                32, 64, 3, 2, padding=1, bias=False
            ),  # [800, 600, 21] -> [400, 300, 11]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            SparseConv3d(
                64, 128, 3, 2, padding=[0, 1, 1], bias=False
            ),  # [400, 300, 11] -> [200, 150, 5]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            SparseConv3d(
                128, 128, (3, 1, 1), (2, 1, 1), bias=False
            ),  # [200, 150, 5] -> [200, 150, 2]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size, input_shape):

        # input: # [41, 1600, 1408]
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]

        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        return ret


@BACKBONES.register_module
class RCNNSpMiddleFHD(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="RCNNSpMiddleFHD", **kwargs
    ):
        super(RCNNSpMiddleFHD, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, bias=False, indice_key="subm0"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(),
            SubMConv3d(16, 16, 3, bias=False, indice_key="subm0"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(),
            SparseConv3d(
                16, 32, 3, 2, padding=1, bias=False
            ),  # [32, 80, 41] -> [16, 40, 21]
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SubMConv3d(32, 32, 3, bias=False, indice_key="subm1"),
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            # SubMConv3d(32, 32, 3, bias=False, indice_key="subm1"),
            # build_norm_layer(norm_cfg, 32)[1],
            # nn.ReLU(),
            SparseConv3d(
                32, 64, 3, 2, bias=False, padding=1
            ),  # [16, 40, 21] -> [8, 20, 11]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, bias=False, indice_key="subm2"),
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            # SubMConv3d(64, 64, 3, bias=False, indice_key="subm2"),
            # build_norm_layer(norm_cfg, 64)[1],
            # nn.ReLU(),
            # SubMConv3d(64, 64, 3, bias=False, indice_key="subm2"),
            # build_norm_layer(norm_cfg, 64)[1],
            # nn.ReLU(),
            SparseConv3d(
                64, 64, 3, 2, bias=False, padding=[1, 1, 0]
            ),  # [8, 20, 11] -> [4, 10, 5]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, bias=False, indice_key="subm3"),
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            # SubMConv3d(64, 64, 3, bias=False, indice_key="subm3"),
            # build_norm_layer(norm_cfg, 64)[1],
            # nn.ReLU(),
            # SubMConv3d(64, 64, 3, bias=False, indice_key="subm3"),
            # build_norm_layer(norm_cfg, 64)[1],
            # nn.ReLU(),
            SparseConv3d(
                64, 64, (1, 1, 3), (1, 1, 2), bias=False
            ),  # [4, 10, 5] -> [4, 10, 2]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size, input_shape):

        # input: # [41, 1600, 1408]
        sparse_shape = np.array(input_shape[::-1]) + [0, 0, 1]

        # coors[:, 1] += 1
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)

        ret = self.middle_conv(ret)

        ret = ret.dense()

        ret = ret.permute(0, 1, 4, 2, 3).contiguous()
        N, C, W, D, H = ret.shape
        ret = ret.view(N, C * W, D, H)

        return ret
