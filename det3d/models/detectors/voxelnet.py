from ..registry import DETECTORS
from .single_stage import SingleStageDetector

@DETECTORS.register_module
class VoxelNet(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        extra_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(VoxelNet, self).__init__(
            reader, backbone, neck, bbox_head, extra_head, train_cfg, test_cfg, pretrained
        )

    def extract_feat(self, data):
        input_features = self.reader(data["features"], data["num_voxels"])
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]
        
        print("example keys: ", example.keys())

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        
        #print("x shape: ", x.shape)
        preds = self.bbox_head(x)
        print("preds: ", len(preds), preds[0]["box_preds"].shape, preds[0]["cls_preds"].shape, preds[4]["box_preds"].shape, preds[4]["cls_preds"].shape)
        
        guided_anchors = self.bbox_head.gg(example, preds)
        
        print("guided anchors: ", len(guided_anchors[0]), len(guided_anchors[1]), len(guided_anchors[0][0]))
        print(guided_anchors[0][0].shape)
        #print("preds: ", len(preds['box_preds']))
        
        bbox_scores = self.extra_head(x, guided_anchors)
        print("bbox score: ", bbox_scores[0].shape)
#         refine_loss_inputs = (bbox_score, ret['gt_bboxes'], ret['gt_labels'], guided_anchors, self.train_cfg.extra)
#         refine_losses = self.extra_head.loss(*refine_loss_inputs)

        if return_loss:
            return self.bbox_head.loss(example, preds)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)
