由于这篇文章是根据DnCNN、IRCNN、BRDNet模型演变而来的，跑实验时候没有给新提出的模型（MBNet）命名，所以在此代码中MBNet模型的代码的文件名为“BRDNet_Cat”。
此代码将MBNet模型的各分支拆开来分析其去噪效果psnr。
代码所用到的常见灰度、彩色数据集在此仓库中都有，分别为：滑铁卢勘探的4744张彩色图数据集，灰度图数据集，train400、Set12、BSD68、CBSD68、Kodak24、Mcmaster等等。