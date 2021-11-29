# Learning to Adapt Structured Output Space for Semantic Segmentation

Pytorch implementation of our method for adapting semantic segmentation from the synthetic dataset (source domain) to the real dataset (target domain). Based on this implementation, our result is ranked 3rd in the [VisDA Challenge](http://ai.bu.edu/visda-2017/).

Contact: Yi-Hsuan Tsai (wasidennis at gmail dot com) and Wei-Chih Hung (whung8 at ucmerced dot edu)

## Paper
[Learning to Adapt Structured Output Space for Semantic Segmentation](https://arxiv.org/abs/1802.10349) <br />
[Yi-Hsuan Tsai](https://sites.google.com/site/yihsuantsai/home)\*, [Wei-Chih Hung](https://hfslyc.github.io/)\*, [Samuel Schulter](https://samschulter.github.io/), [Kihyuk Sohn](https://sites.google.com/site/kihyuksml/), [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/index.html) and [Manmohan Chandraker](http://cseweb.ucsd.edu/~mkchandraker/) <br />
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018 (**spotlight**) (\* indicates equal contribution).



## 复现指标

| iterations/(Batch_size=2) | meanIOU |
| :-----------------------: | :-----: |
|          Target           |  42.35  |
|       Best/(99000)        |  42.7   |
|      latest/(100000)      |  42.32  |
|           5000            |  34.29  |
|           10000           |  37.59  |
|           15000           |  38.84  |
|           20000           |  38.5   |
|           25000           |  39.07  |
|           30000           |  40.52  |
|           35000           |  40.22  |
|           40000           |  39.88  |
|           45000           |  39.88  |
|           50000           |  41.47  |
|           55000           |  41.13  |
|           60000           |  41.74  |
|           65000           |  40.98  |
|           70000           |  42.24  |
|           75000           |  41.47  |
|           80000           |  42.23  |
|           85000           |  42.11  |
|           90000           |  42.14  |
|           95000           |  40.72  |
|                           |         |

## Example Results

![]()

## Quantitative Reuslts

![](figure/iou_comparison_v2.png)

## 数据集

### 1.官网下载

* 下载源域数据集 -GTA5数据集[GTA5 Dataset](https://download.visinf.tu-darmstadt.de/data/from_games/) 。放置在 `data/GTA5` 文件夹下
* 下载目标域数据集-城市景观数据集 [Cityscapes Dataset](https://www.cityscapes-dataset.com/) ,放置在 `data/Cityscapes` 文件夹下

**注意**：

​	（1）GTA5数据集需要全部下载并且解压至同一个文件下，可通过这个[repo](https://github.com/buriedms/Utils.git)当中所提供的`unzips`脚本进行数据集的批量解压

​	（2）下载Cityscapes数据集包括gtFine和leftimg8bit两个数据集，但是仅用到gtFine的验证集部分的灰度图和在leftimg8bit中对应验证集的原始照片和训练照片，可以通过这个[repo](https://github.com/buriedms/Utils.git)当中所提供的`copy_by_txt.py`脚本进行数据集的制作。详细使用图片可通过`dataset/cityscapes_list`当中`train.txt`、`val.txt`、`label.txt`文件进行查看

### 2. 通过aistudio开源数据集获取数据集

​	（1）[GTA5数据集-part1](https://aistudio.baidu.com/aistudio/datasetdetail/106349)

​	（2）[GTA5数据集-part2](https://aistudio.baidu.com/aistudio/datasetdetail/106372)

​	（3）[Cityscapes-valmini](https://aistudio.baidu.com/aistudio/datasetdetail/118666)

### 3.数据集结构目录

```
data
╠═══Citycapes
║   ╚═══data
║       ╠═══gtFine
║       ║   ╠═══test  
║ 		║   ╠═══train  
║		║   ╚═══val  
║		╚══leftimg8bit  
║           ╠═══test  
║ 		    ╠═══train  
║		    ╚═══val  
╚═══GTA5  
    ╠═══images  
	╚═══labels  
```



  

## 预训练模型
* 原始预训练模型[链接]()，可用于模型从头开始训练。
* 150000个iteration模型[链接]()，可用于作为继续训练的预训练模型和持续训练
* 每隔5000iteration的模型[链接]()，提取码：，可用于测试每个阶段的模型效果

## 测试启动
*  LS-GAN and using [Synscapes](https://7dlabs.com/synscapes-overview) as the source domain
  - Performance: check the appendix of the updated [arXiv paper](https://arxiv.org/abs/1802.10349) (updated on 10/17/2019)
  - [Pre-trained models](https://www.dropbox.com/s/sif9cd6ad4s9y5d/AdaptSegNet_LSGAN_models.zip?dl=0)

* Download the pre-trained multi-level [GTA5-to-Cityscapes model](http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth) and put it in the `model` folder

* Test the model and results will be saved in the `result` folder

```
python evaluate_cityscapes.py --restore-from ./model/GTA2Cityscapes_multi-ed35151c.pth
```

* Or, test the VGG-16 based model [Model Link](http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_vgg-ac4ac9f6.pth)

```
python evaluate_cityscapes.py --model DeeplabVGG --restore-from ./model/GTA2Cityscapes_vgg-ac4ac9f6.pth
```


```
0 processd
100 processd
200 processd
300 processd
400 processd
Num classes 19
===>road:	86.46
===>sidewalk:	35.96
===>building:	79.92
===>wall:	23.41
===>fence:	23.27
===>pole:	23.87
===>light:	35.24
===>sign:	14.77
===>vegetation:	83.35
===>terrain:	33.25
===>sky:	75.62
===>person:	58.49
===>rider:	27.55
===>car:	73.65
===>truck:	32.48
===>bus:	35.42
===>train:	3.85
===>motocycle:	30.05
===>bicycle:	28.11
===> mIoU: 42.35
```
* Compute the IoU on Cityscapes (thanks to the code from [VisDA Challenge](http://ai.bu.edu/visda-2017/))
```
python compute_iou.py ./data/Cityscapes/data/gtFine/val result/cityscapes
```

## 训练启动
* **NEW** Train the GTA5-to-Cityscapes model (single-level with LS-GAN)

```
python train_gta2cityscapes_multi.py --snapshot-dir ./snapshots/GTA2Cityscapes_single_lsgan \
                                     --lambda-seg 0.0 \
                                     --lambda-adv-target1 0.0 --lambda-adv-target2 0.01 \
                                     --gan LS
```

* Train the GTA5-to-Cityscapes model (multi-level)

```
python train_gta2cityscapes_multi.py --snapshot-dir ./snapshots/GTA2Cityscapes_multi \
                                     --lambda-seg 0.1 \
                                     --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001
```

* Train the GTA5-to-Cityscapes model (single-level)

```
python train_gta2cityscapes_multi.py --snapshot-dir ./snapshots/GTA2Cityscapes_single \
                                     --lambda-seg 0.0 \
                                     --lambda-adv-target1 0.0 --lambda-adv-target2 0.001
```

## Acknowledgment
Pytorch版本的原仓库：[AdaptSegNet]()。



