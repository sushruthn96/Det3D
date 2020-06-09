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
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(VoxelNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )

    def extract_feat(self, data):
        input_features = self.reader(data["features"], data["num_voxels"])
        x, points = self.backbone(input_features, data["coors"], data["batch_size"], data["input_shape"], data["is_train"])

        if self.with_neck:
            x = self.neck(x)
        return x, points

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)                      

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
            is_train=return_loss,
        )

        x, points = self.extract_feat(data)
        preds = self.bbox_head(x)

        if return_loss:
            return self.bbox_head.loss(example, preds, points)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)
