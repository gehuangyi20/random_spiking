## Instruction of Running the Source Code

### Software Requirements:
The code has been tested on GTX 1080Ti servers. Sample models are provided.

- Tensorflow-gpu (v1.13.1)
- Keras (v2.3.1)
- Numpy (v1.18)
- scipy (v1.1.0)
- easydict matplotlib Pillow

### 1. Environment Setup

```bash
# python3 environment setup
pip3 install --user virtualenv
virtualenv -p python3 nn_defense

# activate environment
source nn_defense/bin/activate

# install required software
pip3 install numpy==1.18
pip3 install scipy==1.1.0
pip3 install tensorflow-gpu==v1.13.1
pip3 install keras==v2.3.1
pip3 install easydict matplotlib Pillow

# exist envirnoment
deactivate
```

### 2. Dataset

- Mnist
```bash
./setup/download_mnist.sh [output dir, default:mnist_data/data]
```

- Fashion
```bash
./setup/download_fashion.sh [output dir, default:fashion_data/data]
```

- Cifar 10 & Cifar 100

The script converts the cifar data into mnist format.
```bash
./setup/download_cifar.sh [output dir, default:cifar_data/data]
./setup/download_cifar100.sh [output dir, default:cifar100_data/data]
```

### 3. Model Training

- Generate Training script

```Bash
python3 RsNet/gen_script_att_and_verify.py --dir=run_script/[dataset]/ -c sample_training.json -o sh_sample_train_
```
> Note ```[dataset]``` can be ```mnist```, ```fashion```, or ```cifar```.

- Train models from scratch
```Bash
bash run_script/[dataset]/sh_sample_train_models.sh
```
The script to train 4 models for each training methods using random selected half dataset.
The sample script provides ```standard``` and ```rs1_nodropout```. 
To train full or split dataset, please use the following configuration file
```run_script/[dataset]/train_[full or split].json```

By default, the script place the program on the first GPU. If you have multiple GPU, you need to change
```--gpu_idx=[idx]``` in the script file in order to place the program running on other GPUs.
The saved model can be found in directory ```result_[dataset]/models_sample```.

- Download sample models
Since training models from scratch takes so much time, we release sample models which
can be downloaded by running following command
```Bash
run_script/sample_download_models.sh [dataset]
```

### 4. Model Accuracy
- Generate model acc verification script
```Bash
python3 RsNet/gen_script_att_and_verify.py --dir=run_script/[dataset]/ -c sample_verify_model.json -o sh_sample_verify_
```

- Evaluate and statistic model accuarcy
```Bash
# run and stat the model acc
bash run_script/[dataset]/sh_sample_verify_models.sh
bash run_script/sample_stat_model_acc.sh [dataset]
```
The result of model accuracy can be found in the directory ```result_[dataset]/models_acc_sample/summary.csv```

### 5. Generate Adversarial Examples
- Initial adversarial examples saving directory and create adversarial examples generation script
```Bash
bash run_script/sample_stat_model_acc.sh [dataset]
```

- Generate the adversarial examples
```Bash
bash run_script/[dataset]/sh_sample_att_single.sh
bash run_script/[dataset]/sh_sample_att_multi.sh
```
Generated adversarial examples can be found in the directory ```[dataset]_data/sample_attack_*```


- The example we provided only generated 10 adversarial example against the target, place the program
on the first GPU. If you want to generate more adversarial examples, you need to modify 
```--num=10``` and ```--start=0``` in file ```run_script/[dataset]/sample_att_[single or multi].json```.
If you want to attack multi-target models and place them on different GPUs, then you need to change
```"--gpu_idx=0,"```. For example, Use ```"--gpu_idx=0,1,2,3"``` if you have 4 GPUs and 4 target models.

### 6. White Box Evaluation
- Run following script to evaluate whitebox attack
```Bash
bash run_script/sample_stat_adv_white_box.sh [dataset]
```

- The evaluation result can be found in the directory ```result_[dataset]/data_sum/sample_[single or multi]```
You may see multiple directory, each corresponding to one ```[attack_name]```. Under each directory, you 
will see 

    - ```summary.csv``` shows the statistic result of C&W attack including attack success rate, confidence value,
$L_p$ distance, and etc
    - ```attack_[target model id]_[confidence].png``` shows the original, adv, and their difference.

### 7. Model Stability
- Initial model stability saving directory and create model stability evaluation script
```Bash
bash run_script/sample_gen_model_stable_script.sh [dataset]
```

- Evaluate model stability
```Bash
# Evaluating model stability with Gaussian Noise
bash run_script/[dataset]/sh_sample_stability_noise.sh
# Evaluating model stability with JPEG compression
bash run_script/[dataset]/sh_sample_stability_noise_jpeg.sh
```

- Stat model stability
```Bash
bash run_script/sample_stat_model_stability.sh [dataset]
```

- The result of model stability can be found in the directory ```result_[dataset]/models_[noise or jpeg]_sample```
    - ```[dataset].csv``` shows the statistic result of model stability for different training methods.
    - ```[dataset]_lg.pdf``` plots the legend used for different training methods.
    - ```[dataset]_plot_unchg.pdf``` plots the effect of noise (Gussian or JPEG compression)
    on prediction stability for each training method

### 8. Gray Box Evaluation
- Verify the transferability of adversarial examples
```Bash
bash run_script/sample_eval_adv_trans_script.sh [dataset]
```

The result can be found in the directory 
```Bash
result_[dataset]/sample_result_cross_[multi or single]/
cross_[target method]/trans_[inference method]_[target model id]
```

For example, ```result_mnist/sample_result_cross_multi/cross_standard/trans_rs1_nodropout_0``` means
that adversarial examples are generated on multiple target models (2 standard target models used here), and
we evaluate how likely those adversarial examples can be transferred to models trained with rs1_nodropout
method. You can find ```standard_transferability_attack_l2_0_[confidence].csv``` and 
```standard_transferability_attack_l2_0_[confidence]-raw.csv``` under the directory, where confidence is
a parameter used in generating adversarial examples. Higher confidence indicates the adversarial example
is more likely to be transferred but also having more distortion.

```standard_transferability_attack_l2_0_[confidence].csv``` shows average transferability and
prediction accuracy on each inference model. ```standard_transferability_attack_l2_0_[confidence]-raw.csv```
shows the $L_p$ value of each adversarial example and whether it can be transferred to the inference model.
Inference model index and original example index are given in first two columns.