# ORNet: Boosting Multi-person 3D Human Pose Estimation by Occluded Joints Reasoning
The code in this file is used to reproduce the results in Table 1 (Comparisons with SOTA methods on MuPoTS-3D), and the ablation results of each module.

Note that the detection module and the reasoning module can be trained together since different losses are added to these two modules separately and the update of the reasoning module does not affect the training of the detection module. In our experiments, the detection module converges first and then the reasoning module converges after about 2-3 epochs.

However, to better reproduce the results, the code we provided here trains these two modules separately, which enables us to monitor the performance of each module step-by-step.

## TODO:

- [ ] Update the code (the current version contains the code we submit before the ECCV submission deadline, you can reproduce the results of the main paper with this code, but some details and implementations are slightly different from the camera-ready paper, and the code is somehow very ugly....... We refined our code and further improved the method after submission. We will update the new version later.)
- [ ] Update training instruction
- [ ] ...

## Environment setup
```bash
# install requirements
pip3 install -r requirements.txt

# install depth-aware part association lib (c++ and cuda)
# requirement: gcc >= 6
cd extensions
# make the cuda path in setup.py right
python setup.py install
```

## Prepare Data for training  
We follow SMAP [1] to prepare the data. Please following their introduction to download and convert all the data.

Put all data under the "data/" folder with symbolic link according to the specified directory structure. 

```
$PROJECT_HOME
|-- data
|   |-- coco2017
|   |   |-- annotations
|   |   |-- train2017
|   |-- MuCo
|   |   |-- annotations
|   |   |-- images
|   |-- ...
```
Then, replace the occlusion labels in "data/MuCo/annotations/MuCo.json" with the labels provided in "occlusion_labels.json"

## Train

```bash
# step 1: train detection module
cd exps/detection
vim train.sh
# change the $PROJECT_HOME to the absolute path of the project
# set $CUDA_VISIBLE_DEVICES and nproc_per_node if using distributed training
bash train.sh
# Model will be saved in "/model_logs/detection" 
# After training, you can evaluate this module following the command provided in the Evaluation section.
# Note that in this step, your model should achieve a PCK of 0.74, approximately. Otherwise, check the occlusion label.


# step 2: train reasoning module
cd exps/reasoning
vim train_infer.sh
# change the $PROJECT_HOME to the absolute path of the project
# set $CUDA_VISIBLE_DEVICES and nproc_per_node if using distributed training
bash train_infer.sh
#Model will be saved in "/model_logs/reasoning" 
# After training, you can evaluate this module following the command provided in the Evaluation section.
# In this step, your model should achieve a PCK of 0.78 at least.


# step 3: generate training data for refinenet
vim config.py
# set DATASET.NAME = 'MIX'
vim refine_infer.sh
# change the $PROJECT_HOME to the absolute path of the project
# set $CUDA_VISIBLE_DEVICES and nproc_per_node if using distributed training
bash refine_infer.sh
# The dataset will be saved in "/model_logs/reasoning" as a json file.


# step 4: train refinenet
cd exps/refinenet
vim config.py
# change the DATA_DIR to the path of the training data you generated in step3
bash train.sh
# Model will be saved in "/model_logs/refine" 
```

## Evaluation

```bash
cd exps/reasoning
vim config.py
# set DATASET.NAME = 'MIX'

vim evaluate_infer.sh
# change the $PROJECT_HOME to the absolute path of the project
# change the path of the trained model; If you only want to evaluate a single model,(e.g. DET or Det+Reason) you can set other path NULL.
bash evaluate_infer.sh
# The results will be saved in "model_logs/reasoning/result/reasoning_generate_result_test_.json" 
```

You need to use an official MATLAB script to evaluate the results. Please follow the introduction provided in [1] to evaluate the results.

[1] Zhen, Jianan, et al. "Smap: Single-shot multi-person absolute 3d pose estimation." *European Conference on Computer Vision*. Springer, Cham, 2020. https://github.com/zju3dv/SMAP
