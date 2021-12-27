# CIFAR-10 Classification
## NonPara HW
此代码基于平常用来跑实验的代码简化而成。

- 利用预训练的ResNet50做fine-tuning，经过3个epoch就可以在测试集达到95%+的正确率

- 直接在ResNet50上做训练，经过100个epoch在测试集可以达到90%正确率（在训练集可达接近100%，模型泛化性稍差）

- 训练在`batch size = 64`时候Linux服务器单张GPU上运行`100 epoch`大约6个小时

### Requirement

- pytorch
- tensorboard
- tqdm

### Dataset
只需修改`run.sh`中`DATA_DIR`变量的值到数据集位置，可以自动下载

### Result
`LOG_DIR`为结果保存根目录，由于是实验代码修改而来，目录层级较多，默认保存在`LOG_DIR/resnet50/classification-cifar10-*/0`之中，目录包含：

- `log.log`文件为训练记录日志
- `events.*`文件为tensorboard记录
- `argv.txt`和`args.json`保存部分超参数
- `checkpoints`保存最后的模型

### Reproduce Result
修改相应数据集和结果保存目录，直接运行命令

``bash run.sh fine_tuning --gpu 0``

进行fine-tuning

``bash run.sh from_scratch --gpu 0``

进行从头开始的训练，具体参数见代码


