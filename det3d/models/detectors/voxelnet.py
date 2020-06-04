from ..registry import DETECTORS
from .single_stage import SingleStageDetector
import torch 
import numpy as np


@DETECTORS.register_module
class VoxelNet(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(VoxelNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )

    def extract_feat(self, data):
        input_features = self.reader(data["features"], data["num_voxels"])
#         print("input features in voxelnet.py", input_features[0])
        x, points = self.backbone(input_features, data["coors"], data["batch_size"], data["input_shape"])
#         print("POINTS[0]", points[0][:5, :])
#         x = self.backbone(input_features, data["coors"], data["batch_size"], data["input_shape"])
        
#         points[0] = points[0][:4]
        
#         x = self.backbone(
#             input_features, data["coors"], data["batch_size"], data["input_shape"]
#         )
        if self.with_neck:
            x = self.neck(x)
        return x, points

    def forward(self, example, return_loss=True, **kwargs):
#         print("KWARGS IN VOXELNET", kwargs)
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]
        
#         for batch_size in range(len(example["annos"])):
#             for i in range(6):
#                 print("example annos",example["annos"][batch_size]["gt_boxes"][i].shape)
#             print("REG TARGETS LAST 5", example["reg_targets"][i][0,:-5,:])
        batch_size = len(num_voxels)
#         for i in range(6):
#             print("EXAMPLE0!!!",len(example["labels"][i][0]==0))
#             print("EXAMPLE-1!!!",len(example["labels"][i][0]==-1))
#             print("EXAMPLE1!!!",len(example["labels"][i][0]==1))
#             print("EXAMPLE2!!!",len(example["labels"][i][0]==2))
                        

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x, points = self.extract_feat(data)
#         print("POINTS TUPLE", points)
        preds = self.bbox_head(x)

        if return_loss:
            return self.bbox_head.loss(example, preds, points)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)
