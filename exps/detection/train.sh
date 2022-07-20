export CUDA_VISIBLE_DEVICES=" "
export PROJECT_HOME=' ' # replace it with right path
export PYTHONPATH=$PYTHONPATH:$PROJECT_HOME
python3 config.py -log
python3 -m torch.distributed.launch --nproc_per_node=8 train.py 