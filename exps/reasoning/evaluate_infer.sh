export CUDA_VISIBLE_DEVICES=" "
export PROJECT_HOME=' '
export PYTHONPATH=$PYTHONPATH:$PROJECT_HOME
python test_infer_for_evaluate_ori-Refine.py \
-p "Path to Detection model" \
-t generate_result \
-d test \
-ip "Path to Reasoning model" \
--batch_size 16 \
--do_flip 1 \
--dataset_path "../../test_img/MultiPersonTestSet" \
-rp "Path to Refinement model"
