# Det3D

Data and annotation creation commands
* create data and annotation database - python tools/create_data.py nuscenes_data_prep --root_path=NUSCENES_TRAINVAL_DATASET_ROOT --version="v1.0-trainval" --nsweeps=10

Train command
* train - # python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py examples/cbgs/configs/nusc_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py --work_dir=$NUSC_CBGS_WORK_DIR

Test command
* test - python -m torch.distributed.launch --nproc_per_node=8 ./tools/dist_test.py $CONFIG --work_dir=$WORK_DIR --checkpoint=$CHECKPOINT
