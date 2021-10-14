import torch
import torch.nn as nn

class conv_block (nn.Module):
    def __init__ (self, in_channels, out_channels, kernel_size, stride,
                  padding):
        super(conv_block, self).__init__()

        self.conv = nn.Conv2d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = kernel_size,
                              stride = stride,
                              padding=padding)

        self.bn = nn.BatchNorm2d(num_features = out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):

        out = self.leaky_relu(self.bn(self.conv(x)))

        return out

class yolov1 (nn.Module):
    def __init__(self, in_channels=3, split_size=7, num_boxes=2, num_classes=20):
        super(yolov1, self).__init__()

        self.first_layer = conv_block(in_channels=3,
                                      out_channels=64,
                                      kernel_size=(7,7),
                                      stride=(2,2),
                                      padding=(3,3))

        self.maxpool = nn.MaxPool2d(kernel_size=(2,2),
                                    stride=(2, 2),
                                    padding=(0,0))

        self.second_layer = conv_block(in_channels=64,
                                       out_channels=192,
                                       kernel_size=(3,3),
                                       stride=(1,1),
                                       padding=(1,1))

        self.third_layer1 = conv_block(in_channels=192,
                                       out_channels=128,
                                       kernel_size=(1,1),
                                       stride=(1,1),
                                       padding=(0,0))

        self.third_layer2 = conv_block(in_channels=128,
                                       out_channels=256,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding=(1, 1))

        self.third_layer3 = conv_block(in_channels=256,
                                       out_channels=256,
                                       kernel_size=(1, 1),
                                       stride=(1, 1),
                                       padding=(0, 0))

        self.third_layer4 = conv_block(in_channels=256,
                                       out_channels=512,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding=(1, 1))

        self.fourth_layer1 = self._create_layer(in_channels=512,
                                               out_channels1=256,
                                               out_channels2=512,
                                               repeats=4)

        self.fourth_layer2 = conv_block(in_channels=512,
                                        out_channels=512,
                                        kernel_size=(1,1),
                                        stride=(1,1),
                                        padding=(0,0))

        self.fourth_layer3 = conv_block(in_channels=512,
                                        out_channels=1024,
                                        kernel_size=(3, 3),
                                        stride=(1, 1),
                                        padding=(1, 1))

        self.fifth_layer1 = self._create_layer(in_channels=1024,
                                              out_channels1=512,
                                              out_channels2=1024,
                                              repeats=2)

        self.fifth_layer2 = conv_block(in_channels=1024,
                                       out_channels=1024,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding=(1, 1))

        self.fifth_layer3 = conv_block(in_channels=1024,
                                       out_channels=1024,
                                       kernel_size=(3, 3),
                                       stride=(2, 2),
                                       padding=(1, 1))

        self.fc1 = nn.Linear(in_features=1024 * split_size * split_size,
                             out_features=4096)

        self.dropout = nn.Dropout(p=0.5)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.fc2 = nn.Linear(in_features=4096,
                             out_features= split_size * split_size *
                                           (num_classes + num_boxes * 5))

    def _create_layer (self, in_channels, out_channels1, out_channels2,
                       repeats):
        layers = []

        for _ in range(repeats):
            layers.append(conv_block(in_channels=in_channels,
                                     out_channels=out_channels1,
                                     kernel_size=(1,1),
                                     stride=(1, 1),
                                     padding=(0,0)))

            layers.append(conv_block(in_channels=out_channels1,
                                     out_channels=out_channels2,
                                     kernel_size=(3,3),
                                     stride=(1, 1),
                                     padding=(1, 1)))

        return nn.Sequential(*layers)


    def forward(self, x):

        out = self.maxpool(self.first_layer(x))
        out = self.maxpool(self.second_layer(out))
        out = self.third_layer1(out)
        out = self.third_layer2(out)
        out = self.third_layer3(out)
        out = self.maxpool(self.third_layer4(out))
        out = self.fourth_layer1(out)
        out = self.fourth_layer2(out)
        out = self.maxpool(self.fourth_layer3(out))
        out = self.fifth_layer1(out)
        out = self.fifth_layer2(out)
        out = self.fifth_layer3(out)
        ##
        out = self.fifth_layer2(out)
        out = self.fifth_layer2(out)

        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out)
        out = self.leaky_relu(self.dropout(out))
        out = self.fc2(out)

        return out

x = torch.randn([1,3,448,448])
model = yolov1(in_channels=3, split_size=7, num_boxes=2, num_classes=20)
print(model(x).shape)