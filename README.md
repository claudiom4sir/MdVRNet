# MdVRNet: Deep Video Restoration under Multiple Distortions


## Overview
This code is the Pytorch implementation of the paper "MdVRNet: Deep Video Restoration under Multiple Distortions" (VISAPP 2022).

### Abstract
Video restoration techniques aim to remove artifacts, such as noise, blur, and compression, introduced at various levels within and outside the camera imaging pipeline during video acquisition. 
Although excellent results can be achieved by considering one artifact at a time, in real applications a given video sequence can be affected by multiple artifacts, whose appearance is mutually influenced.
In this paper, we present Multi-distorted Video Restoration Network (MdVRNet), a deep neural network specifically designed to handle multiple distortions simultaneously.
Our model includes an original Distortion Parameter Estimation sub-Network (DPEN) to automatically infer the intensity of various types of distortions affecting the input sequence, novel Multi-scale Restoration Blocks (MRB) to extract complementary features at different scales using two parallel streams, and implements a two-stage restoration process to focus on different levels of detail.
We document the accuracy of the DPEN module in estimating the intensity of multiple distortions, and present an ablation study that quantifies the impact of the DPEN and MRB modules. Finally, we show the advantages of the proposed MdVRNet in a direct comparison with another existing state-of-the-art approach for video restoration.

### Architecture
![](https://github.com/claudiom4sir/MdVRNet/blob/main/images/mdvrnet.png)
### Results
![](https://github.com/claudiom4sir/MdVRNet/blob/main/images/results.png)

## Datasets
In the paper, we used the following datasets:
- *DAVIS 2017*: 120 480p sequences
- *Set8*: 4 sequences from the *Derf 480p* testset ("tractor", "touchdown", "park_joy", "sunflower") plus other 4 540p sequences
### Trainset
We trained MdVRNet using the [*DAVIS 2017* trainset](https://www.dropbox.com/sh/20n4cscqkqsfgoj/AACfjXp3q6tW-S56l_noKzO3a/training?dl=0&subfolder_nav_tracking=1).
### Testsets
We evaluated MdVRNet using the [*DAVIS 2017* testset](https://drive.google.com/file/d/1seZVrqSlbx89fd43FOQUk0YVli64hEe1/view?usp=sharing) and the [*Set8* dataset](https://www.dropbox.com/sh/20n4cscqkqsfgoj/AABGftyJuJDwuCLGczL-fKvBa/test_sequences?dl=0&subfolder_nav_tracking=1). 

## User guide

### Dependencies
Python 3.6 + CUDA 11.2
- torch==1.2.0 
- torchvision==0.2.1
- scikit-image==0.16.2
- pytest==5.4.1
- pycodestyle==2.5.0
- opencv-python==3.4.2.17
- future==0.18.2
- tensorboardx==2.0
- nvidia-dali==0.10.0

You can install all the python dependencies by executing
```
pip install -r requirements.txt
```
### Help
All the scripts necessary for training and testing the models have an helper function that shows all the possible parameters you can specify in the script itself. To visualize such parameters, add ```--h``` when you call the scripts. In the following, only the basic commands are specified.
### Training
#### DPEN
First of all, you need to train DPEN on single images to recognize the intensity of the artifacts (sigma for AWGN and q for JPEG compression). To train it, execute
```
python train_dpen.py --trainset_dir <trainset_dir> --valset_dir <valset_dir> --sigma <min_sigma> <max_sigma> --q <min_q> <max_q>
```
The trainset and validationset directories are expected to follow this format
```
root_dir/
  |-- seq1/
    |-- im1.png
    |-- img2.png
    |-- ...
  |-- seq2/
    |-- im1.png
    |-- img2.png
    |-- ...
  |-- ...
```
If you want to use the DAVIS 2017 trainset, which contains videos in .mp4 format, you can obtain the aforementioned folder structure by executing
```
python generate_png_from_mp4.py --input_dir <dir_containing_.mp4_files> --output_dir <output_dir>
```
Note that ```generate_png_from_mp4.py``` requires [FFmpeg](https://www.ffmpeg.org/), make sure it is installed before running the script.
#### MdVRNet
Once DPEN is trained, you can train MdVRNet on video sequences by executing
```
python train_mdvrnet.py --trainset_dir <trainset_dir> --log_dir <log_dir> --sigma <min_sigma> <max_sigma> --q <min_q> <max_q> --DPEN_model <DPEN_model>.pth
```
The trainset directory is expected to follow this format
```
trainset_dir/
  |-- seq1.mp4
  |-- seq1.mp4
  |-- ...
```

**Note**: To speed up the training process of MdVRNet, we used the [DALI](https://developer.nvidia.com/dali) library, which requires input sequences to be in a video format (.mp4 to be precise). If your data are sequences of images, you can generate videos in .mp4 format using [FFmpeg](https://www.ffmpeg.org/). The DALI library is used only for training, while for testing you can use sequences of images.

### Testing
The DPEN and MdVRNet pretrained models (trained on DAVIS 2017) are available [here](https://github.com/claudiom4sir/MdVRNet/tree/main/pretrained_models).
#### DPEN
You can test a pretrained DPEN model by executing
```
python test_dpen.py --DPEN_model <DPEN_model>.pth --valset_dir <valset_dir> --sigma <sigma> --q <q>
```
The testset directory is expected to follow the same format as in training
#### MdVRNet
You can test a pretrained MdVRNet model (pretrained DPEN model is required) by executing
```
python test_mdvrnet.py --model_file <MdVRNet_model>.pth --test_path <test_dir> --noise_sigma <sigma> --q <q> --DPEN_model <DPEN_model>.pth --save_path <out_dir>
```
The testset directory is expected to contain only a sequence and to follow this format
```
test_dir/
  |-- im1.png
  |-- im2.png
  |-- ...
```

## Citations
If you think this project is useful for your research, please cite our paper
```
TODO
```

## Acknowledgements
The code is based on the excellent work done by [Tassano et al.](https://github.com/m-tassano/fastdvdnet).

## Contacts
For any question, please write an email to c.rota30@campus.unimib.it
