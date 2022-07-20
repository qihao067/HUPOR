export CUDA_VISIBLE_DEVICES=" "
export PROJECT_HOME=' ' # replace it with your path
export PYTHONPATH=$PYTHONPATH:$PROJECT_HOME
python config.py -log
python -m torch.distributed.launch --nproc_per_node=8 train_infer.py