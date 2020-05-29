import torch.nn as nn
import numpy as np
from mmdet.models.utils import one_hot
from mmdet.ops.iou3d import iou3d_utils
from mmdet.ops.iou3d.iou3d_utils import boxes3d_to_bev_torch
import torch
import torch.nn.functional as F
from mmdet.core.loss.losses import weighted_smoothl1, weighted_sigmoid_focal_loss, weighted_cross_entropy
from mmdet.core.utils.misc import multi_apply
from mmdet.core.bbox3d.target_ops import create_target_torch
import mmdet.core.bbox3d.box_coders as boxCoders
from mmdet.core.post_processing.bbox_nms import rotate_nms_torch
from functools import partial
from ..registry import EXTRA_HEAD

def gen_sample_grid(box, window_size=(4, 7), grid_offsets=(0, 0), spatial_scale=1.):
    N = box.shape[0]
    win = window_size[0] * window_size[1]
    xg, yg, wg, lg, rg = torch.split(box, 1, dim=-1)

    xg = xg.unsqueeze_(-1).expand(N, *window_size)
    yg = yg.unsqueeze_(-1).expand(N, *window_size)
    rg = rg.unsqueeze_(-1).expand(N, *window_size)

    cosTheta = torch.cos(rg)
    sinTheta = torch.sin(rg)

    xx = torch.linspace(-.5, .5, window_size[0]).type_as(box).view(1, -1) * wg
    yy = torch.linspace(-.5, .5, window_size[1]).type_as(box).view(1, -1) * lg

    xx = xx.unsqueeze_(-1).expand(N, *window_size)
    yy = yy.unsqueeze_(1).expand(N, *window_size)

    x=(xx * cosTheta + yy * sinTheta + xg)
    y=(yy * cosTheta - xx * sinTheta + yg)

    x = (x.permute(1, 2, 0).contiguous() + grid_offsets[0]) * spatial_scale
    y = (y.permute(1, 2, 0).contiguous() + grid_offsets[1]) * spatial_scale

    return x.view(win, -1), y.view(win, -1)


def bilinear_interpolate_torch_gridsample(image, samples_x, samples_y):
    C, H, W = image.shape
    image = image.unsqueeze(1)  # change to:  C x 1 x H x W

    samples_x = samples_x.unsqueeze(2)
    samples_x = samples_x.unsqueeze(3)
    samples_y = samples_y.unsqueeze(2)
    samples_y = samples_y.unsqueeze(3)

    samples = torch.cat([samples_x, samples_y], 3)
    samples[:, :, :, 0] = (samples[:, :, :, 0] / (W - 1))  # normalize to between  0 and 1
    samples[:, :, :, 1] = (samples[:, :, :, 1] / (H - 1))  # normalize to between  0 and 1
    samples = samples * 2 - 1  # normalize to between -1 and 1

    return torch.nn.functional.grid_sample(image, samples)

@EXTRA_HEAD.register_module
class PSWarpHead(nn.Module):
    def __init__(self, grid_offsets, featmap_stride, in_channels, num_class=1, num_parts=49):
        super(PSWarpHead, self).__init__()
        self._num_class = num_class
        out_channels = num_class * num_parts

        self.gen_grid_fn = partial(gen_sample_grid, grid_offsets=grid_offsets, spatial_scale=1 / featmap_stride)

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=False)
        )

    def forward(self, x, guided_anchors, is_test=False):
        x = self.convs(x)
        bbox_scores = list()
        for i, ga in enumerate(guided_anchors):
            if len(ga) == 0:
                bbox_scores.append(torch.empty(0).type_as(x))
                continue
            (xs, ys) = self.gen_grid_fn(ga[:, [0, 1, 3, 4, 6]])
            im = x[i]
            out = bilinear_interpolate_torch_gridsample(im, xs, ys)
            score = torch.mean(out, 0).view(-1)
            bbox_scores.append(score)

        if is_test:
            return bbox_scores, guided_anchors
        else:
            return torch.cat(bbox_scores, 0)


    def loss(self, cls_preds, gt_bboxes, gt_labels, anchors, cfg):

        batch_size = len(anchors)

        labels, targets, ious = multi_apply(create_target_torch,
                                            anchors, gt_bboxes,
                                            (None,) * batch_size, gt_labels,
                                            similarity_fn=getattr(iou3d_utils, cfg.assigner.similarity_fn)(),
                                            box_encoding_fn = second_box_encode,
                                            matched_threshold=cfg.assigner.pos_iou_thr,
                                            unmatched_threshold=cfg.assigner.neg_iou_thr)

        labels = torch.cat(labels,).unsqueeze_(1)

        # soft_label = torch.clamp(2 * ious - 0.5, 0, 1)
        # labels = soft_label * labels.float()

        cared = labels >= 0
        positives = labels > 0
        negatives = labels == 0
        negative_cls_weights = negatives.type(torch.float32)
        cls_weights = negative_cls_weights + positives.type(torch.float32)

        pos_normalizer = positives.sum().type(torch.float32)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        cls_targets = labels * cared.type_as(labels)
        cls_preds = cls_preds.view(-1, self._num_class)

        cls_losses = weighted_sigmoid_focal_loss(cls_preds, cls_targets.float(), \
                                                 weight=cls_weights, avg_factor=1.)

        cls_loss_reduced = cls_losses / batch_size

        return dict(loss_cls=cls_loss_reduced,)

    def get_rescore_bboxes(self, guided_anchors, cls_scores, img_metas, cfg):
        det_bboxes = list()
        det_scores = list()

        for i in range(len(img_metas)):
            bbox_pred = guided_anchors[i]
            scores = cls_scores[i]

            if scores.numel == 0:
                det_bboxes.append(None)
                det_scores.append(None)

            bbox_pred = bbox_pred.view(-1, 7)
            scores = torch.sigmoid(scores).view(-1)
            select = scores > cfg.score_thr

            bbox_pred = bbox_pred[select, :]
            scores = scores[select]

            if scores.numel() == 0:
                det_bboxes.append(bbox_pred)
                det_scores.append(scores)
                continue

            #boxes_for_nms = boxes3d_to_bev_torch(bbox_pred)
            keep = rotate_nms_torch(bbox_pred, scores, iou_threshold=cfg.nms.iou_thr)

            bbox_pred = bbox_pred[keep, :]
            scores = scores[keep]

            det_bboxes.append(bbox_pred.detach().cpu().numpy())
            det_scores.append(scores.detach().cpu().numpy())

        return det_bboxes, det_scores
    
@EXTRA_HEAD.register_module
class MultiGroupPSWarpHead(nn.Module):
    def __init__(self, grid_offsets, featmap_stride, in_channels, num_classes=[1,2,2,1,2,2], num_groups=6, num_parts=49):
        super(MultiGroupPSWarpHead, self).__init__()
        
        self.grid_offsets = grid_offsets
        self.featmap_stride = featmap_stride
        self.in_channels = in_channels
        self.num_groups = num_groups
        self.num_classes = num_classes
        self.num_parts = num_parts
        
        print("num parts ", self.num_parts)
        self.warp_heads = nn.ModuleList()
        for i in range(self.num_groups):
            self.warp_heads.append(
                PSWarpHead(
                    grid_offsets, 
                    featmap_stride, 
                    in_channels, 
                    self.num_classes[i], 
                    num_parts=self.num_parts
                )
            )
            
    def forward(self, x, guided_anchors, is_test=False):
        bbox_scores = []
        for i, warp_head in enumerate(self.warp_heads):
            bbox_scores.append(warp_head(x, guided_anchors[i], is_test))

        return bbox_scores
            
    def loss(self, cls_preds, gt_bboxes, gt_labels, anchors, cfg):
        self.warp_losses = []
        for i in range(self.num_groups):
            self.warp_losses.append( self.warp_heads[i].loss(cls_preds[i],
                                                             gt_bboxes[i], 
                                                             gt_labels[i], 
                                                             anchors[i], 
                                                             cfg) )            
    
        