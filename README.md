# AlphaGo-Zero-gobang
基于AlphaGo Zero的多层次自训练
代码共包括七个.py文件，分别为：
MCTS.py: 包含两个类：1.class MCTS，蒙特卡洛树的主类，并在类中定义了模拟、搜索相关的函数。
                                    2.class Node，蒙特卡洛树节点类，包含了节点的子节点信息以及Q、U等信息。
Board.py: 包含一个类class Board，定义了棋盘状态，保存了当前棋盘信息和玩家信息。
Axial_resnet.py: 定义了Axial_resnet深度神经网络
Net.py: 定义了Easy_model深度神经网络
resnet_18.py: 定义了resnet_18深度神经网络
PSO.py: 定义了粒子群优化网络，用于优化蒙特卡洛树的决策。
train.py: 用于定义机器自我博弈并训练网络以及人机对弈的函数。
效果。。。还有问题
