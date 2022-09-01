import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
# from IRCNN import DnCNN
from utils import *
from middle_and_lower import BRDNet
# from IRCNN_9 import DnCNN

import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=20, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs/middle_and_lower_patch5050-4744-S-25", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68 from gray image,'
                                                                    'test on Kodak24 or CBSD68 or McMaster from color image')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
parser.add_argument("--test_epochs", type=int, default=50, help="Number of test epochs")
parser.add_argument("--channels", type=int, default=1, help='Number of input channels')
parser.add_argument("--color", type=bool, default=False, help="color or gray")
opt = parser.parse_args()

def normalize(data):
    return data/255.

def main():

    Total_Max_PSNR = 0
    start_time = time.time()
    for i in range(30, 51):     ### 导入 net 1~50.pth 权重

        str_1 = 'net ' + str(i) + '.pth'
        # Build model
        print('\nLoading model ...' + str_1)

        # net = DnCNN(channels=opt.channels)     ####  运行IRCNN时，应该把  num_of_layers=opt.num_of_layers  删除
        net = BRDNet(channels=opt.channels)
        # model = net.cuda()
        device_ids = [0]
        model = nn.DataParallel(net, device_ids=device_ids).cuda()
        model.load_state_dict(torch.load(os.path.join(opt.logdir, str_1)))  #### 加载权重
        # model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net 40.pth')))     #### 加载权重
        print('\nLoading model ...finished\n')

        Max_PSNR = 0

        for x in range(opt.test_epochs):   ###测试50轮test 的 PSNR的值，保存最大值

            model.eval()   ##  验证模型
            # load data info
            print('Loading data info ...\n')             ### 数据信息
            files_source = glob.glob(os.path.join('data', opt.test_data, '*.png' or '*.bmp' or '*.jpg'))
            files_source.sort()
            # process data

            psnr_test = 0
            # ssim_test = 0


            for f in files_source:
                # image
                Img = cv2.imread(f)

                # Img = np.transpose(Img, (2, 0, 1))            ##  彩色图
                # Img = normalize(np.float32(Img))
                # Img = np.expand_dims(Img, 0)
                #
                # # TODO
                # # Img = normalize(np.float32(Img[:,:,0]))          ## 灰度图
                # # Img = np.expand_dims(Img, 0)
                # # Img = np.expand_dims(Img, 1)
                if opt.color:         ## 彩色图处理
                    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
                    Img = np.transpose(Img, (2, 0, 1))
                    Img = normalize(np.float32(Img))
                    Img = np.expand_dims(Img, 0)
                    # print(Img.shape)
                else:                  ## 灰度图处理
                    Img = normalize(np.float32(Img[:,:,0]))
                    Img = np.expand_dims(Img, 0)
                    Img = np.expand_dims(Img, 1)


                # clean image
                ISource = torch.Tensor(Img)
                # noise
                noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
                # noisy image
                INoisy = ISource + noise
                # move to GPU
                ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())



                with torch.no_grad(): # this can save much memory
                    Out = torch.clamp(INoisy-model(INoisy), 0., 1.)

                ## if you are using older version of PyTorch, torch.no_grad() may not be supported
                # ISource, INoisy = Variable(ISource.cuda(),volatile=True), Variable(INoisy.cuda(),volatile=True)
                # Out = torch.clamp(INoisy-model(INoisy), 0., 1.)

                psnr = batch_PSNR(Out, ISource, 1.)
                # ssim = structural_similarity(Out, ISource,multichannel=True)

                psnr_test += psnr
                # ssim_test += ssim

                print(" ”%s “  PSNR : %f    " % (f, psnr))
                # print(" ”%s “  PSNR : %f   SSIM :%f  " % (f, psnr, ssim))

            psnr_test /= len(files_source)
            # ssim_test /= len(files_source)

            # 记录每轮上的psnr的平均值
            print("\n***************~~~~~~~~**************第 %d 轮    PSNR on test data: %.3f" % (x+1,psnr_test))
            # print("\n***************~~~~~~~~**************第 %d 轮    PSNR on test data: %.3f" % (i + 1, ssim_test))

            if psnr_test > Max_PSNR:
                Max_PSNR = psnr_test
                j = x+1
            # if ssim_test > Max_SSIM:
            #   Max_SSIM = ssim_test

        # 记录每个 net *.pth 上测试50轮上的psnr的最大值
        print("\n********************************************************第 %d 轮 "% j + "Max_PSNR on    "+str_1+"   test data: %.3f" % Max_PSNR)
        # print("\n*******************************************~~~~~~~~~~~~~~~~~~*************  Max_SSIM on test data: %.3f" % Max_SSIM)



        if Max_PSNR > Total_Max_PSNR:
            Total_Max_PSNR = Max_PSNR
            str_2 = str_1
            k = j
    ## 记录全部net *.pth 上,最大的psnr值
    print("\n*******************************************~~~~~~~~~~~~~~~~~~*************第 %d 轮 "% k  + "Total_Max_PSNR on   "+str_2+"  test data: %.3f " % Total_Max_PSNR )
    # # print("\n*******************************************~~~~~~~~~~~~~~~~~~*************  Max_SSIM on test data: %.3f" % Max_SSIM)
    end2_time = time.time()
    print("\n*****************************   时间消耗: %.3f 小时" % ((end2_time-start_time)/3600))

if __name__ == "__main__":
    main()
