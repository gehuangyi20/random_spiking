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

# make RsNet to be accessible in current environment
ln -s ../../../../RsNet/ nn_defense/lib/python3.6/site-packages/RsNet

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

    The script converts the cifar data into mnist dataset format.
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
    The script train 4 models for each training method with randomly selected half dataset.
    The sample script provides two training methods ```standard``` and ```rs1_nodropout```. 
    To train full or split dataset, please use the following configuration file
    ```run_script/[dataset]/train_[full or split].json```
    
    By default, the script places the program on the first GPU. If you have multiple GPUs, you need to change
    ```--gpu_idx=[idx]``` in the script file in order to place the program running on other GPUs.
    The saved model can be found in the directory ```result_[dataset]/models_sample```.

- Download **sample models**

    Since training models from scratch takes so much time, we release sample models which
    can be downloaded by running the following command.
    ```Bash
    run_script/sample_download_models.sh [dataset]
    ```
    Each sample model is trained with randomly selected half dataset.

### 4. Model Accuracy
- Generate model acc verification script
    ```Bash
    python3 RsNet/gen_script_att_and_verify.py --dir=run_script/[dataset]/ -c sample_verify_model.json -o sh_sample_verify_
    ```

- Evaluate and statistic model accuarcy
    ```Bash
    # run and stat the model acc
    bash run_script/[dataset]/sh_sample_verify_models.sh
    run_script/sample_stat_model_acc.sh [dataset]
    ```
    The result of model accuracy can be found in the directory ```result_[dataset]/models_acc_sample/summary.csv```

### 5. Generate Adversarial Examples
- Initial adversarial examples saving directory and create adversarial examples generation script
    ```Bash
    run_script/sample_gen_adv_script.sh [dataset]
    ```

- Generate the adversarial examples
    ```Bash
    bash run_script/[dataset]/sh_sample_att_single.sh
    bash run_script/[dataset]/sh_sample_att_multi.sh
    ```

    Generated adversarial examples can be found in the directory ```[dataset]_data/sample_attack_*```

- The example we provided only generated 10 adversarial example against the target model. The 
sample program runs on the first GPU. If you want to generate more adversarial examples, you need to modify 
```--num=10``` and ```--start=0``` in the file ```run_script/[dataset]/sample_att_[single or multi].json```.
If you want to attack multi-target models and place the program on different GPUs, then you need to change
```"--gpu_idx=0,"```. For example, Use ```"--gpu_idx=0,1,2,3"``` if you have 4 GPUs and 4 target models.

- **Adversarial example file format**

    For those who want to further work on generated adversarial examples, we present the file format of
generated Adversarial examples. All file having ```*.gz``` extension is compressed with GZip.

    - ```[att_name]-adv-img.gz```
    Raw generated adversarial images are encoded in standard ```[batch, height, width, channels]``` format.
    Each pixel is stored in ```float32``` format.
    
    - ```[att_name]-adv-label.gz```
    Targeted label of Raw generated adversarial images are encoded in ```[batch, label]``` format. Each
    label is stored in ```uint8``` format.
    
    - ```[att_name]-adv-raw-label.gz```
    Original label of Raw generated adversarial images are encoded in ```[batch, label]``` format. Each
    label is stored in ```uint8``` format.
    
    - ```[att_name]-img.gz```
    For each given original image, we store the alternative format of raw generated adversarial images
    (```[att_name]-adv-img.gz```) into floor, ceiling, and round format. The binary file is encoded as 
    ```<16 zero bytes>+[(floor_img, ceil_img, round_img), ..., (floor_img, ceil_img, round_img)]```.
    Each adversarial example is encoded in standard ```[height, width, channels]```format. Each pixel
    is stored in ```uint8``` format.
    
    - ```[att_name]-label.gz```
    Given the adversarial examples in ```[att_name]-adv-img.gz```, we store the prediction output 
    of targeted model. The binary file is encoded as ```<8 zero bytes>+[(floor_label, ceil_label, 
    round_label), ..., (floor_label, ceil_label, round_label)]```. Each label is stored in
    ```uint8``` format.
    
    - ```[att_name]-idx.pkl```
    Index of original image in the dataset is stored as ```[label, label, ..., label]``` in an numpy array.
    The array is serialized in python pickle format.

### 6. White Box Evaluation
- Run following script to evaluate whitebox attack
    ```Bash
    bash run_script/sample_stat_adv_white_box.sh [dataset]
    ```

- The evaluation result can be found in the directory ```result_[dataset]/data_sum/sample_[single or multi]```
You may see multiple directory, each corresponding to one ```[attack_method]```. Under each directory, you 
will see 

    - ```summary.csv``` shows the statistic result of C&W attack including attack success rate, confidence value,
$L_p$ distance, and etc
    - ```attack_[target model id]_[confidence].png``` shows the original, adv, and their difference.
    - ```sum_attack_[target model id]_[confidence].txt``` shows the prediction output label of the target model.

### 7. Model Stability
- Initial model stability saving directory and create model stability evaluation script
    ```Bash
    run_script/sample_gen_model_stable_script.sh [dataset]
    ```

- Evaluate model stability
    ```Bash
    # Evaluating model stability with Gaussian Noise
    bash run_script/[dataset]/sh_sample_stability_noise.sh
    # Evaluating model stability with JPEG compression
    bash run_script/[dataset]/sh_sample_stability_noise_jpeg.sh
    ```

- Statistic model stability
    ```Bash
    run_script/sample_stat_model_stability.sh [dataset]
    ```

- The result of model stability can be found in the directory ```result_[dataset]/models_[noise or jpeg]_sample```
    - ```[dataset].csv``` shows the statistic result of model stability for different training methods.
    - ```[dataset]_lg.pdf``` plots the legend used for different training methods.
    - ```[dataset]_plot_unchg.pdf``` plots the effect of noise (Gussian or JPEG compression)
    on prediction stability for each training method
    
    The sample script only tests on the first 100 images. If you want to cover the whole dataset,
    you need to modify ```--test_data_start=0 --test_data_len=100``` in the file
    ```run_script/[dataset]/sample_stability_[noise or jpeg].json```.

### 8. Gray Box Evaluation
- Verify the transferability of adversarial examples
    ```Bash
    run_script/sample_eval_adv_trans_script.sh [dataset]
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
    shows the $L_p$ distance of each adversarial example and whether it can be transferred to the inference model.
    Inference model index and original example index are given in the first two columns.
