# CalibOnline
This method aims to provide an online miscalibration detection and correction scheme for calibrated multi-sensor systems that require long-term operation.

Multisensor fusion is increasingly used in robotic systems due to its improvement in system robustness and accuracy, whitch also poses great challenges for Multisensor calibration. Existing techniques are mainly implemented offline and with the help of targets, which cannot cope with extrinsic perturbation caused by vibrations and deformations while the system is running. In this letter, we present CalibOnline, a novel method for online detecting and correcting extrinsic perturbation. First, this letter introduces a unified data modality for representing LiDAR and cameras, i.e., the depth map, and a robust feature that this modality possesses, i.e., depth discontinuous edges, is explored. Secondly, the effect of extrinsic perturbation on the edge-matching constraints is analyzed and accordingly the miscalibration probabilities are designed to supervise the extrinsic. Finally, the perturbation correction is described as a problem of on-manifolds optimization, which enhances the convergence of the estimated extrinsic. The experimental results in different datasets and scenarios have demonstrated that the proposed method has high robustness and accuracy.

![The pipeline of CalibOnline](https://github.com/cchester25/CalibOnline/blob/main/pipeline.gif)

## 1. Prerequisites
### 1.1 **Ubuntu** and **ROS**
Ubuntu 64-bit 20.04.
ROS noetic. [ROS Installation](http://wiki.ros.org/ROS/Installation) or [One click installation](https://fishros.org.cn/forum/topic/20/%E5%B0%8F%E9%B1%BC%E7%9A%84%E4%B8%80%E9%94%AE%E5%AE%89%E8%A3%85%E7%B3%BB%E5%88%97?lang=en-GB)
```bash
    sudo apt-get install ros-XXX-cv-bridge ros-xxx-pcl-conversions
```
or
```bash
wget http://fishros.com/install -O fishros && . fishros
```
### 1.2 **Eigen 3.3.7**
Follow [Eigen Installation](http://eigen.tuxfamily.org/index.php?title=Main_Page)

### 1.3 **Ceres Solver 1.14.0**
Follow [Ceres Installation](http://ceres-solver.org/installation.html).

### 1.4 **virtual environment **
conda create -n depth python=3.8
conda activate depth
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
### Pretrained models
[**[Models]**](https://1drv.ms/u/s!AiV6XqkxJHE2mULfSmi4yy-_JHSm?e=s97YRM) 
### 1.5 **data set **
[kitti_2011-09-30-0027](https://pan.baidu.com/s/1ghUL_sVwFSTMDg1Pei84dQ?pwd=9wxo)
```bash
pip install kitti2bag
kitti2bag -t 2011_10_03 -r 0042 raw_synced
```

## 2. Build
Clone the repository and catkin_make:

```bash
cd ~/catkin_ws/src
git clone https://github.com/cchester25/CalibOnline.git
cd ../
catkin_make
source ~/catkin_ws/devel/setup.bash
```
## 3. Run example
```bash
cd depthV3
conda activate depth
roscore
python inference.py --config configs/v3/kitti_raw.txt --ckpt_path ckpts/kitti_scv3/xxxxx.ckpt 
```
```bash
cd ~/catkin_ws/src
source ~/catkin_ws/devel/setup.bash
roslaunch calib_online run_outdoor_kitti.launch
```
## 👏 Acknowledgement

This project is based on the following projects:
- [SC-DepthV3](https://github.com/JiawangBian/sc_depth_pl.git) -Dynamic Scene Depth Estimation Method
- [ip_basic](https://github.com/kujason/ip_basic.git) - Sparse Depth Completion
