# MdVRNet: Deep Video Restoration under Multiple Distortions


## Overview
This code is the Pytorch implementation of the paper "MdVRNet: Deep Video Restoration under Multiple Distortions" (VISAPP 2022).

### Abstract
Video restoration techniques aim to remove artifacts, such as noise, blur, and compression, introduced at various levels within and outside the camera imaging pipeline during video acquisition. 
Although excellent results can be achieved by considering one artifact at a time, in real applications a given video sequence can be affected by multiple artifacts, whose appearance is mutually influenced.
In this paper, we present Multi-distorted Video Restoration Network (MdVRNet), a deep neural network specifically designed to handle multiple distortions simultaneously.
Our model includes an original Distortion Parameter Estimation sub-Network (DPEN) to automatically infer the intensity of various types of distortions affecting the input sequence, novel Multi-scale Restoration Blocks (MRB) to extract complementary features at different scales using two parallel streams, and implements a two-stage restoration process to focus on different levels of detail.
We document the accuracy of the DPEN module in estimating the intensity of multiple distortions, and present an ablation study that quantifies the impact of the DPEN and MRB modules. Finally, we show the advantages of the proposed MdVRNet in a direct comparison with another existing state-of-the-art approach for video restoration.

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

You can install all the python dependencies executing
```
pip install -r requirements.txt
```
### Training
First of all, you need to train DPEN on single images to recognize the intensity of the artifacts (sigma for AWGN and q for JPEG compression). To train it, execute
```
TODO
```
Once DPEN is trained, you can train MdVRNet on video sequences executing
```
python train_mdvrnet.py --trainset_dir <trainset_dir> --log_dir <log_dir> --estimate_parameter_model <DPEN_model>.pth
```
For more training options, execute
```
python train_mdvrnet.py --help
```
This will show you, for example, how to change the range of the artifacts, the number of epochs, the batch size, the patch size etc.

The trainset directory is expected to follow this format
```
trainset_dir/
  |-- seq1.mp4
  |-- seq1.mp4
  |-- ...
```

#### Note
To speed up the training process of MdVRNet, we used the [DALI](https://developer.nvidia.com/dali) library, which requires input sequences to be in a video format (.mp4 to be precise). If your data are sequences of images, you can generate videos in .mp4 format using [FFmpeg](https://www.ffmpeg.org/). The DALI library is used only for training, while for testing you can use sequences of images.

### Testing
Once you have trained MdVRNet, you can test it executing
```
python test_mdvrnet.py --model_file <MdVRNet_model>.pth --test_path <test_dir> --noise_sigma <sigma> --q <q> --estimate_parameter_model <DPEN_model>.pth --save_path <out_dir>
```
The testset directory is expected to contain only a sequence and to follow this format
```
test_dir/
  |-- im1.png
  |-- im2.png
  |-- ...
```
The DPEN and MdVRNet pretrained models (trained on DAVIS 2017) are available [TODO]().
## Citations
If you think this project is useful for your research, please cite our paper
```
TODO
```

## Acknowledgements
The code is based on the excellent work done by [Tassano et al.](https://github.com/m-tassano/fastdvdnet)

## Contacts
For any question, please write an email to c.rota30@campus.unimib.it
