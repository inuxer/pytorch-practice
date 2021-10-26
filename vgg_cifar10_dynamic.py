import torchvision
import torch
import os
import torch.utils.data as Data
import numpy as np
from torchvision import transforms
from torch import nn
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

BATCH_SIZE = 200
EPOCH = 100
LR = 0.1

cfgAll = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}

DOWNLOAD_CIFAR10 = False
if not(os.path.exists('./CIFAR10/')) or not os.listdir('./CIFAR10/'):
    DOWNLOAD_CIFAR10 = True

train_data = torchvision.datasets.CIFAR10(
    root='./CIFAR10',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_CIFAR10
)

print(train_data.data.shape)
print(train_data.targets.__len__())
# plt.imshow(train_data.data[0])
# plt.show()


train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, transform=transforms.ToTensor())
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = self.vgg16_bn(cfgAll, True)

        self.out = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        #print(x.size)
        output = self.out(x)
        return output

    def vgg16_bn(self, cfgAll, batch_norm):
        return VGG.make_layers(cfgAll['D'], batch_norm)

    #动态生成指定的的卷积层
    def make_layers(cfg, batch_norm):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)






gpus = [0]
cuda_gpu = torch.cuda.is_available()
vgg = VGG()  #VGG类的初始化，生成相应的神经网络
#vgg = VGG().vgg16_bn(cfgAll, batch_norm=True)


if cuda_gpu:
    vgg = torch.nn.DataParallel(vgg, device_ids=gpus).cuda()


optimizer = torch.optim.SGD(vgg.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


time_start = time.time()
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        if cuda_gpu:
            b_x = b_x.cuda()
            b_y = b_y.cuda()
        output = vgg(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            LR *= 0.99
            accuracy = 0
            for test_step, (test_x, test_y) in enumerate(test_loader):
                if cuda_gpu:
                    test_x = test_x.cuda()
                test_out = vgg(test_x)
                pred_y = torch.max(test_out, 1)[1].data
                if cuda_gpu:
                    pred_y = pred_y.cpu().numpy()
                else:
                    pred_y = pred_y.numpy()
                accuracy += float((pred_y == np.array(test_y)).astype(int).sum()) / float(len(test_y))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data, '| test accuracy: %.2f' % (accuracy/(test_step+1)))

time_end = time.time()
print("all time consume: ", time_end-time_start)
