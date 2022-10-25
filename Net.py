import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Easy_model(nn.Module):
    def __init__(self, input_layer):
        super(Easy_model, self).__init__()
        self.conv1 = nn.Conv2d(input_layer, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

    def forward(self, in_put):
        in_put = self.relu(self.bn1(self.conv1(in_put)))
        in_put = self.relu(self.bn2(self.conv2(in_put)))
        in_put = self.relu(self.bn3(self.conv3(in_put)))

        return in_put


class Model(nn.Module):
    def __init__(self, input_layer, board_size):
        super(Model, self).__init__()
        # self.model = resnet18(input_layers=input_layer)
        # self.p = para[board_size]
        self.model = Easy_model(input_layer)
        self.p = 11
        self.output_channel = 128
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        # value head
        self.value_conv1 = nn.Conv2d(kernel_size=1, in_channels=self.output_channel, out_channels=16)
        self.value_bn1 = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(in_features=16 * self.p * self.p, out_features=256)
        self.value_fc2 = nn.Linear(in_features=256, out_features=1)
        # policy head
        self.policy_conv1 = nn.Conv2d(kernel_size=1, in_channels=self.output_channel, out_channels=16)
        self.policy_bn1 = nn.BatchNorm2d(16)
        self.policy_fc1 = nn.Linear(in_features=16 * self.p * self.p, out_features=board_size * board_size)

    def forward(self, state):
        s = self.model(state)

        # value head part
        v = self.value_conv1(s)
        v = self.relu(self.value_bn1(v)).view(-1, 16 * self.p * self.p)
        v = self.relu(self.value_fc1(v))
        value = self.tanh(self.value_fc2(v))

        # policy head part
        p = self.policy_conv1(s)
        p = self.relu(self.policy_bn1(p)).view(-1, 16 * self.p * self.p)
        prob = self.policy_fc1(p)
        return prob, value


class neuralnetwork:
    def __init__(self, input_layers, board_size, use_cuda=None, learning_rate=0.01):
        self.use_cuda = use_cuda
        if use_cuda:
            self.model = Model(input_layer=input_layers, board_size=board_size).cuda().double()
        else:
            self.model = Model(input_layer=input_layers, board_size=board_size)
        self.opt = torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=learning_rate, weight_decay=1e-4)

        self.mse = nn.MSELoss()
        self.crossloss = nn.CrossEntropyLoss()

    def train(self, data_loader):
        self.model.train()
        loss_record = []
        for batch_idx, (state, strategy, winner) in enumerate(data_loader):
            tmp = []
            state, strategy, winner = Variable(state), Variable(strategy), Variable(winner)
            state = state.to(torch.float32)
            strategy = strategy.to(torch.float32)
            winner = winner.to(torch.float32)
            if self.use_cuda:
                state, strategy, winner = state.cuda(), strategy.cuda(), winner.cuda()
            self.opt.zero_grad()
            prob, value = self.model(state)
            output = F.log_softmax(prob, dim=1)

            # loss1 = F.kl_div(output, distrib)
            # loss2 = F.mse_loss(value, winner)
            # loss1.backward(retain_graph=True)
            # loss2.backward()
            cross_entropy = - torch.mean(torch.sum(strategy * output, 1))
            mse = F.mse_loss(value, winner)
            # loss = F.mse_loss(value, winner) - torch.mean(torch.sum(distrib*output, 1))
            loss = cross_entropy + mse
            loss.backward()

            self.opt.step()
            tmp.append(loss.data)
            if batch_idx % 10 == 0:
                print("We have played and batch {}, the cross entropy loss is {}, the mse loss is {}".format(
                   batch_idx, cross_entropy.data, mse.data))
                loss_record.append(sum(tmp) / len(tmp))
        return loss_record

    def eval(self, state):
        self.model.eval()
        if self.use_cuda:
            state = torch.from_numpy(state).unsqueeze(0).double().cuda()
        else:
            # state = torch.from_numpy(state).unsqueeze(0).double()
            state = torch.from_numpy(state).unsqueeze(0).float()
        with torch.no_grad():
            prob, value = self.model(state)
        return F.softmax(prob, dim=1), value

    def adjust_lr(self, lr):
        for group in self.opt.param_groups:
            group['lr'] = lr


