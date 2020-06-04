from .points_ops import *
from mmdet.ops.points_op import points_op_cpu
import torch

def pts_in_boxes3d(pts, boxes3d):
    N = len(pts)
    M = len(boxes3d)
    pts_in_flag = torch.IntTensor(M, N).fill_(0)
    reg_target = torch.FloatTensor(N, 3).fill_(0)
#     print("pts in flag", pts_in_flag)
#     print("reg_target", reg_target)
    points_op_cpu.pts_in_boxes3d(pts.contiguous(), boxes3d.contiguous(), pts_in_flag, reg_target)
#     print("pts in flag", torch.unique(pts_in_flag[:,i]) for i in range(N))
#     li = []
#     for i in range(N):
#         li.append(torch.unique(pts_in_flag[:,i]))
#     print(li)
#     print(torch.unique(pts_in_flag))
#     print("reg_target", reg_target)
    return pts_in_flag, reg_target

