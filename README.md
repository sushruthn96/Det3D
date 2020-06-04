# Structure Aware Det3D

## Installation
Please see [INSTALLATION.md](https://github.com/sushruthn96/Det3D/blob/master/INSTALLATION.md) for detailed installation instructions.

## Training 
First prepare dataset for training:
```shell
python tools/create_data.py nuscenes_data_prep --root_path=$NUSCENES_TRAINVAL_DATASET_ROOT --version="v1.0-trainval" --nsweeps=10
```
Then initiate training with following command:
```shell
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS ./tools/train.py examples/cbgs/configs/nusc_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py --work_dir=$WORK_DIR
```

## Inference
First prepare dataset for testing:
```shell
python tools/create_data.py nuscenes_data_prep --root_path=$NUSCENES_TRAINVAL_DATASET_ROOT --version="v1.0-test" --nsweeps=10
```
Testing with trained model weights can be carried out as follows:
```shell
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS ./tools/dist_test.py $CONFIG --work_dir=$WORK_DIR --checkpoint=$CHECKPOINT
```
