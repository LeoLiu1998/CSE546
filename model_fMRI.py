import torch
import torch.nn as nn
import torch.nn.functional as F

class CBR(nn.Module):  #same size
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-03)
        self.act = nn.ReLU(True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output

class DownSampler_1d(nn.Module):
    def __init__(self, nIn, nOut, kSize=3):
        super().__init__()
        self.conv = Con_1d(nIn, nOut, kSize)
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-03)
        self.act = nn.ReLU(True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output

class DownSamplerA(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.conv = CBR(nIn, nOut, 3, 2)

    def forward(self, input):
        output = self.conv(input)
        return output

class Con_1d(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=(2, 1)):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=(0, padding), bias=False)

    def forward(self, input):
        output = self.conv(input)
        return output

class CDilated(nn.Module): #same size
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False,
                              dilation=d)

    def forward(self, input):
        output = self.conv(input)
        return output

class BR(nn.Module): #same size
    def __init__(self, nOut):
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-03)
        self.act = nn.ReLU(True)  # nn.PReLU(nOut)

    def forward(self, input):
        output = self.bn(input)
        output = self.act(output)
        return output

class CB(nn.Module):  #same size
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-03)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return output

class DilatedParllelResidualBlockB1(nn.Module):  # with k=4
    def __init__(self, nIn, nOut, prob=0.03):
        super().__init__()
        k = 4
        n = int(nOut / k)
        n1 = nOut - (k - 1) * n
        self.c1 = Con_1d(nIn, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-3)
        self.act = nn.ReLU(True)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)

        add1 = d1 + d2
        add2 = add1 + d4
        add3 = add2 + d8

        combine = torch.cat([d1, add1, add2, add3], 1)
        combine_in_out = input + combine
        output = self.bn(combine_in_out)
        output = self.act(output)
        return output




class ResNetC1(nn.Module):
    def __init__(self,):
        super().__init__()
        self.level1 = CBR(51, 64, 3, 1)
        self.level1_0 = CBR(64, 64, 3, 1)

        self.level2 = DownSampler_1d(64, 64)
        self.level2_0 = DilatedParllelResidualBlockB1(64, 64)
        self.level2_1 = DilatedParllelResidualBlockB1(64, 64)

        self.br_2 = BR(128)

        self.level3_0 = DownSamplerA(128, 64)
        self.level3_1 = DownSamplerA(64, 32)

        # self.level3_0 = CBR(128, 64, 3, 1)
        # self.level3_1 = DilatedParllelResidualBlockB1(64, 64, 0.3)
        # self.level3_2 = DilatedParllelResidualBlockB1(64, 64, 0.3)
        #
        # self.level4_0 = CBR(192, 128, 3, 1)
        # self.level4_1 = DilatedParllelResidualBlockB1(128, 128, 0.3)
        # self.level4_2 = DilatedParllelResidualBlockB1(128, 128, 0.3)
        #
        # self.br_4 = BR(192)
        # self.br_con_4 = BR(256)
        #
        # self.level5_0 = CBR(256, 64, 3, 1)
        # self.level5_1 = DilatedParllelResidualBlockB1(64, 64, 0.3)
        # self.level5_2 = DilatedParllelResidualBlockB1(64, 64, 0.3)
        #
        # self.br_con_5 = BR(128)

        self.global_Avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(32, 24)
        self.fc2 = nn.Linear(24, 12)
        self.logSoftmax = nn.LogSoftmax()


    def forward(self, input1): #(1, 5, 174, 18)
        # input1 = self.cmlrn(input)
        output0 = self.level1(input1)
        output1_0 = self.level2(output0)
        output1 = self.level2_0(output1_0)
        output1 = self.level2_1(output1)

        output1 = self.br_2(torch.cat([output1_0, output1], 1))
        print(output1.shape)
        output1 = self.level3_0(output1)
        output1 = self.level3_1(output1)
        #
        # output2_0 = self.level3_0(output1)
        # output2 = self.level3_1(output2_0)
        # output2 = self.level3_2(output2)
        #
        # output2 = self.br_2(torch.cat([output2_0, output2], 1))
        #
        # output3 = self.level4_1(output2)
        # output3 = self.level4_2(output3)
        #
        # output3 = self.br_4(torch.cat([output2_0, output3], 1))
        #
        # l5_0 = self.level4_0(output3)
        # l5_1 = self.level4_1(l5_0)
        # l5_2 = self.level4_2(l5_1)
        # l5_con = self.br_con_4(torch.cat([l5_0, l5_2], 1))
        #
        # l6_0 = self.level5_0(l5_con)
        # l6_1 = self.level5_1(l6_0)
        # l6_2 = self.level5_2(l6_1)
        # l6_con = self.br_con_5(torch.cat([l6_0, l6_2], 1))

        print(output1.shape)
        glbAvg = self.global_Avg(output1)
        flatten = glbAvg.view(glbAvg.size(0), -1)
        fc1 = self.fc1(flatten)
        output = self.fc2(fc1)
        prob = self.logSoftmax(output, axis=1)


        return prob


# Introduce an extra dimension in dataloader first!!!
class Conv3dNet(nn.Module):
    def __init__(self, ninput, noutput, kernel_size, stride=1, num_classes=12):
        super(Conv3dNet, self).__init__()
        self.padding = int((kernel_size - 1) / 2)
        self.layer1 = nn.Sequential(
            nn.Conv3d(ninput, noutput, kernel_size, stride=stride, padding=self.padding),
            nn.BatchNorm3d(noutput),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(noutput, 2 * noutput, kernel_size, stride=stride, padding=self.padding),
            nn.BatchNorm3d(noutput * 2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(32 * 12 * 14 * 5, num_classes)


    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.fc(out.view(len(out), -1))
        return out


class Conv2dNet(nn.Module):
    def __init__(self, ninput, noutput, kernel_size, stride=1, num_classes=12):
        super(Conv2dNet, self).__init__()
        self.padding = int((kernel_size - 1) / 2)
        self.layer1 = nn.Sequential(
            nn.Conv2d(ninput, noutput, kernel_size, stride=stride, padding=self.padding),
            nn.BatchNorm2d(noutput),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(noutput, 32, kernel_size, stride=stride, padding=self.padding),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(32 * 14 * 5, num_classes)


    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.fc(out.view(len(out), -1))
        return out

class FC(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.fc = nn.Linear(nIn, nOut)

    def forward(self, input):
        output = self.fc(input)
        return output

net = Conv2dNet(51, 64, kernel_size=2)
a = torch.Tensor(1, 51, 61, 23)
out1 =net(a)
print(out1.size())