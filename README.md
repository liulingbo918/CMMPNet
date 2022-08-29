# Aerial Images Meet Crowdsourced Trajectories: A New Approach to Robust Road Extraction


In this work, we focus on a challenging task of land remote sensing analysis, i.e., automatic extraction of traffic roads from remote sensing data. Nevertheless, conventional methods either only utilized the limited information of aerial images, or simply fused multimodal information (e.g., vehicle trajectories), thus cannot well recognize unconstrained roads. To facilitate this problem, we introduce a novel neural network framework termed Cross-Modal Message Propagation Network, which fully benefits the complementary different modal data (i.e., aerial images and crowdsourced trajectories). Extensive experiments on three real-world benchmarks demonstrate the effectiveness of our method for robust road extraction benefiting from blending different modal data, either using image and trajectory data or image and Lidar data.


If you use this code for your research, please cite [our paper](https://ieeexplore.ieee.org/abstract/document/9696168) :

```
@article{liu2022aerial,
  title={Aerial images meet crowdsourced trajectories: a new approach to robust road extraction},
  author={Liu, Lingbo and Yang, Zewei and Li, Guanbin and Wang, Kuo and Chen, Tianshui and Lin, Liang},
  journal={IEEE transactions on neural networks and learning systems},
  year={2022},
  publisher={IEEE}
}
```
## Requirements
```
numpy
opencv_python
scikit_learn
torch
torchvision
tqdm
```

## Dataset Preprocessing
download the following datasets and put them into the folder  ```dataset/```.
- BJRoad: The original dataset (including satellite images and vehicle trajectories) can be requested from the author of [CVPR2019 paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Leveraging_Crowdsourced_GPS_Data_for_Road_Extraction_From_Aerial_Imagery_CVPR_2019_paper.pdf). They done [data augmentation](https://github.com/suniique/Leveraging-Crowdsourced-GPS-Data-for-Road-Extraction-from-Aerial-Imagery/blob/b174045888b9b7daf2c61b03fa6f922048480863/utils/data_loader.py#L57) during the training period. Here we provide the data-augmented satellite images and trajectory heatmaps (resolution: 1024*1024) used in our work. \[[Google.Drive](https://drive.google.com/file/d/1LwTn8_wpsLRBuYW7w6pmxSIhdVNGcze5/view?usp=sharing)\]   \[[BaiduYun, password：hiwv](https://pan.baidu.com/s/1kfbw0SKoQqNoG08mM-KGMA)\]

- Porto: This dataset contains 6,048 pairs of satellite images and trajectory heatmaps with a resolution of 512*512. We conduct five-fold
cross-validation experiments on this dataset. \[[Google.Drive](https://drive.google.com/file/d/1L3uqySCaIwoa-U22LTqKRemxlHhfKZL7/view?usp=sharing)\]   \[[BaiduYun, password：ffia](https://pan.baidu.com/s/1_mkVOnoTr_wxrK00t3Ac5Q)\]

- TLCGIS: This is a  road extraction dataset with 5,860 pairs of satellite images and Lidar images. Their resolution is 500*500. In this dataset, the label of foreground road is 0. \[[Download](  http://ww2.cs.fsu.edu/~parajuli/datasets/fusion_lidar_images_sigspatial18.zip)\]



## Training and Testing
```bash
# experiment on BJRoad
sh train_val_test_BJRoad.sh

# experiment on Porto
sh train_val_test_Porto.sh

# experiment on TLCGIS
sh train_val_test_TLCGIS.sh
```
## Notes
1)  In the [source code](https://github.com/suniique/Leveraging-Crowdsourced-GPS-Data-for-Road-Extraction-from-Aerial-Imagery/blob/master/framework.py#L106) of the [CVPR2019 paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Leveraging_Crowdsourced_GPS_Data_for_Road_Extraction_From_Aerial_Imagery_CVPR_2019_paper.pdf), average IoU is the mean of the IoU of all batches (2 GPUs, 4 samples/GPU). For a fair comparison, we follow this work to compute the average IoU in our paper. However, more strictly, average_iou should be the mean of the IoU of all samples. So I recommend using global IoU.

2) In our experiments, we did not use data augmentation for Porto and TLCGIS. Their performance can be further improved when using data augmentation (i.e., setting the parameter [randomize](https://github.com/liulingbo918/CMMPNet/blob/main/utils/datasets.py#L57) to True).

