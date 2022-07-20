export CUDA_VISIBLE_DEVICES=" "
export PROJECT_HOME=' '
export PYTHONPATH=$PYTHONPATH:$PROJECT_HOME
python test_infer_for_evaluate_ori-Refine.py \
-p "Path to checkpoint of detection model" \
-t generate_train \
-d generation \
-ip "Path to checkpoint of reasoning model" \
--batch_size 16 \
--do_flip 1
