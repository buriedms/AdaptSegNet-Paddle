# AdaptSegNet：学习调整结构化输出空间进行语义分割

英文名：Learning to Adapt Structured Output Space for Semantic Segmentation

将语义分割从合成数据集（源域）调整到真实数据集（目标域）的方法的PaddlePaddle实现。

点击[此处](https://aistudio.baidu.com/aistudio/projectdetail/3056745?shared=1)可跳转aistudio项目，直接运行项目。

## Paper
[Learning to Adapt Structured Output Space for Semantic Segmentation](https://arxiv.org/abs/1802.10349) <br />
[Yi-Hsuan Tsai](https://sites.google.com/site/yihsuantsai/home)\*, [Wei-Chih Hung](https://hfslyc.github.io/)\*, [Samuel Schulter](https://samschulter.github.io/), [Kihyuk Sohn](https://sites.google.com/site/kihyuksml/), [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/index.html) and [Manmohan Chandraker](http://cseweb.ucsd.edu/~mkchandraker/) <br />
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018 (**spotlight**) (\* indicates equal contribution).


### 摘要
基于卷积神经网络的语义分割方法依赖于像素级地面真实度的监督，但可能无法很好地推广到看不见的图像域。由于标记过程繁琐且劳动密集，因此开发能够将源地真相标记适应目标域的算法非常有意义。在本文中，提出了一种基于语义分割的领域自适应对抗学习方法。考虑到语义分割是包含源域和目标域之间空间相似性的结构化输出，我们在输出空间中采用了对抗式学习。为了进一步增强自适应模型，我们构建了一个多级对抗网络，在不同的特征层次上有效地执行输出空域自适应。在各种领域适应设置下进行了广泛的实验和研究，包括合成到真实和跨城市场景。

### 网络结构
1. 鉴别器。对于鉴别器，该网络由5个卷积层组成，核数为4×4，步长为2，其中通道数分别为{64、128、256、512、1}。除最后一层外，每个卷积层后面都有一个泄漏的ReLU，参数为0.2。由于使用了小批量与分段网络联合训练鉴别器，因此不使用任何批次标准化层。
2. 分割网络。为了获得高质量的分割结果，必须建立良好的基线模型。我们采用DeepLab-v2框架，并在ImageNet上预先训练ResNet-101模型作为分割基线网络。
3. 多层次适应模型。我们构建了上述鉴别器和分割网络作为我们的单级自适应模型。对于多层结构，我们从Conv4层提取特征映射，并添加一个ASPP模块作为辅助分类器。类似地，为对抗性学习添加了具有相同体系结构的鉴别器。在本文中，由于其效率和准确性的平衡，我们使用了两个级别。


## 复现指标及训练过程 

| iterations/(Batch_size=2) | meanIOU | iterations/(Batch_size=2) | meanIOU |
| :-----------------------: | :-----: | :-----------------------: | :-----: |
|          Target           |  42.35  |           Best            |  42.72  |
|           5000            |  34.29  |           10000           |  37.59  |
|           15000           |  38.84  |           20000           |  38.5   |
|           25000           |  39.07  |           30000           |  40.52  |
|           35000           |  40.22  |           40000           |  39.88  |
|           45000           |  39.88  |           50000           |  41.47  |
|           55000           |  41.13  |           60000           |  41.74  |
|           65000           |  40.98  |           70000           |  42.24  |
|           75000           |  41.47  |           80000           |  42.23  |
|           85000           |  42.11  |           90000           |  42.14  |
|           95000           |  40.72  |          100000           |  41.58  |
|          105000           |  40.82  |          110000           |  42.72  |
|          115000           |  40.96  |          120000           |  39.86  |
|          125000           |  42.15  |          130000           |  40.72  |
|          135000           |  42.09  |          140000           |  41.1   |
|          145000           |  41.67  |          150000           |  40.19  |

## Example Results

![img1](figure/result.png)

## Quantitative Reuslts

![img2](figure/iou_comparison_v2.png)

## 数据集

### 1.官网下载

* 下载源域数据集 -GTA5数据集[GTA5 Dataset](https://download.visinf.tu-darmstadt.de/data/from_games/) 。放置在 `data/GTA5` 文件夹下
* 下载目标域数据集-城市景观数据集 [Cityscapes Dataset](https://www.cityscapes-dataset.com/) ,放置在 `data/Cityscapes` 文件夹下

**注意**：

​	（1）GTA5数据集需要全部下载并且解压至同一个文件下，可通过这个[repo](https://github.com/buriedms/Utils)当中所提供的`unzips.py`脚本进行数据集的批量解压

​	（2）下载Cityscapes数据集包括gtFine和leftimg8bit两个数据集，但是仅用到gtFine的验证集部分的灰度图和在leftimg8bit中对应验证集的原始照片和训练照片，
可以通过这个[repo](https://github.com/buriedms/Utils)当中所提供的`copy_by_txt.py`脚本进行数据集的制作。详细使用图片可通过`dataset/cityscapes_list`当中`train.txt`、`val.txt`、`label.txt`文件进行查看

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
║       ║   ╠═══train  
║       ║   ╚═══val  
║       ╚═══leftimg8bit  
║           ╠═══test  
║           ╠═══train  
║           ╚═══val  
╚═══GTA5  
    ╠═══images  
    ╚═══labels  
```

## 训练模型和日志
* 原始预训练模型[链接](https://aistudio.baidu.com/aistudio/datasetdetail/119256)，可用于模型从头开始训练。
* 已训练Best模型[链接](https://aistudio.baidu.com/aistudio/datasetdetail/119256)，可用于作为继续训练的预训练模型和持续训练
* 每阶段模型参数和训练日志[链接](https://aistudio.baidu.com/aistudio/datasetdetail/119256)，可用于测试每个阶段的模型效果

## 测试启动
- 下载测试模型(可选Best模型、各阶段模型、中途训练模型 ）并且放置在`model`路径下
* 使用以下代码测试模型，并且模型将会保存在`result`文件夹下（restore-from：模型文件路径）

```
python evaluate_cityscapes.py --restore-from ./model/GTA2Cityscapes_multi-ed35151c.pth 
```

* 测试结果案例：


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
* 如果您想单独对已生成的result结果计算iou，可使用例如以下代码。(thanks to the code from [VisDA Challenge](http://ai.bu.edu/visda-2017/))
```
python compute_iou.py ./data/Cityscapes/data/gtFine/val result/cityscapes
```

## 训练启动

* **重新训练**GTA5-to-Cityscapes模型 (multi-level)

```
python train_gta2cityscapes_multi.py --checkpoint-dir ./checkpoint/GTA2Cityscapes_multi \
                                     --lambda-seg 0.1 \
                                     --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001
```

* **继续训练**GTA5-to-Cityscapes模型 (multi-level)

```
python train_gta2cityscapes_multi.py --checkpoint-dir ./checkpoint/GTA2Cityscapes_multi \
                                     --lambda-seg 0.1 \
                                     --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001 \
                                     --start-iter latest \
                                     --continue-train 
```

- 重点参数解释

| 重点参数       | 含义                                                         |
| -------------- | ------------------------------------------------------------ |
| checkpoint-dir | 模型结果及日志保存位置                                       |
| continue-train | 是否启用持续学习策略（触发有效）                             |
| start-iter     | 持续学习开始的iter数，默认为latest，即从上次保存点开始，启动持续学习时有效 |


**注意：**训练日志存放在`checkpoint-dir`目录下

## Acknowledgment
Pytorch版本的原仓库：[AdaptSegNet](https://github.com/wasidennis/AdaptSegNet)。



