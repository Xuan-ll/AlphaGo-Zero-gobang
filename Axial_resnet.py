import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch
from operator import itemgetter


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))


def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)


# calculates the permutation to bring the input tensor to something attend-able
# also calculates the inverse permutation to bring the tensor back to its original shape

def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]

    permutations = []

    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)

    return permutations


class PermuteToFrom(nn.Module):
    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def forward(self, x, **kwargs):
        axial = x.permute(*self.permutation).contiguous()

        shape = axial.shape
        *_, t, d = shape

        # merge all but axial dimension
        axial = axial.reshape(-1, t, d)

        # attention
        axial = self.fn(axial, **kwargs)

        # restore to original shape and permutation
        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial


# attention

class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads=None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias=False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv=None):
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out


# axial attention class

class AxialAttention(nn.Module):
    def __init__(self, dim, num_dimensions=2, heads=8, dim_heads=None, dim_index=-1, sum_axial_out=True):
        assert (dim % heads) == 0, 'hidden dimension must be divisible by number of heads'
        super().__init__()
        self.dim = dim
        self.total_dimensions = num_dimensions + 2
        self.dim_index = dim_index if dim_index > 0 else (dim_index + self.total_dimensions)

        attentions = []
        for permutation in calculate_permutations(num_dimensions, dim_index):
            attentions.append(PermuteToFrom(permutation, SelfAttention(dim, heads, dim_heads)))

        self.axial_attentions = nn.ModuleList(attentions)
        self.sum_axial_out = sum_axial_out

    def forward(self, x):
        assert len(x.shape) == self.total_dimensions, 'input tensor does not have the correct number of dimensions'
        assert x.shape[self.dim_index] == self.dim, 'input tensor does not have the correct input dimension'

        if self.sum_axial_out:
            return sum(map(lambda axial_attn: axial_attn(x), self.axial_attentions))

        out = x
        for axial_attn in self.axial_attentions:
            out = axial_attn(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.Axial_attention = AxialAttention(dim = planes, dim_index = 1, dim_heads = 32, heads = 1, num_dimensions = 2, sum_axial_out = True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + self.Axial_attention(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Axial_ResNet(nn.Module):

    def __init__(self, block, layers, input_layer):
        super(Axial_ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(input_layer, 16, kernel_size=3, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


def axial_resnet18(input_layers):
    model = Axial_ResNet(block=BasicBlock, layers=[3, 3, 3], input_layer=input_layers)
    return model


class Model(nn.Module):
    def __init__(self, input_layer, board_size):
        super(Model, self).__init__()
        self.model = axial_resnet18(input_layers=input_layer)
        self.output_channel = 64
        self.p = 5
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
        v = self.relu(self.value_bn1(v))
        v = torch.flatten(v, 1)
        v = self.relu(self.value_fc1(v))
        value = self.tanh(self.value_fc2(v))

        # policy head part
        p = self.policy_conv1(s)
        p = self.relu(self.policy_bn1(p))
        p = torch.flatten(p, 1)
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
        # self.opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

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
