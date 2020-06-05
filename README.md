# Structure Aware Det3D

## Installation
Please see [INSTALLATION.md](https://github.com/sushruthn96/Det3D/blob/master/INSTALLATION.md) for detailed installation instructions.

## Training 
First prepare dataset for training:
```shell
python tools/create_data.py nuscenes_data_prep --root_path=$NUSCENES_TRAINVAL_DATASET_ROOT --version="v1.0-trainval" --nsweeps=10
```
Next, modify the config file:
```python
dataset_type = "NuScenesDataset"
n_sweeps = 10
data_root = "/data/Datasets/nuScenes"
```
Specify task and anchor:
**The order of tasks and anchors must be the same**

```python
tasks = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]
anchor_generators=[
    dict(
        type="anchor_generator_range",
        sizes=[1.97, 4.63, 1.74],
        anchor_ranges=[-50.4, -50.4, -0.95, 50.4, 50.4, -0.95],
        rotations=[0, 1.57],
        velocities=[0, 0],
        matched_threshold=0.6,
        unmatched_threshold=0.45,
        class_name="car",
    ),
    dict(
        type="anchor_generator_range",
        sizes=[2.51, 6.93, 2.84],
        anchor_ranges=[-50.4, -50.4, -0.40, 50.4, 50.4, -0.40],
        rotations=[0, 1.57],
        velocities=[0, 0],
        matched_threshold=0.55,
        unmatched_threshold=0.4,
        class_name="truck",
    ),
    ...
]
```
Finally, initiate training with following command:
```shell
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS ./tools/train.py $CONFIG_FILE --work_dir=$WORK_DIR
```

## Inference
First prepare dataset for testing:
```shell
python tools/create_data.py nuscenes_data_prep --root_path=$NUSCENES_TRAINVAL_DATASET_ROOT --version="v1.0-test" --nsweeps=10
```
Testing with trained model weights can be carried out as follows:
```shell
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS ./tools/dist_test.py $CONFIG_FILE --work_dir=$WORK_DIR --checkpoint=$CHECKPOINT
```
