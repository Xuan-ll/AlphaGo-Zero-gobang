import numpy as np
import torch
import torch.utils.data as torch_data
from Board import Board
from Axial_resnet import neuralnetwork as nn
from MCTS import MCTS
from PSO import PSO
import matplotlib.pyplot as plt


class random_queue:
    def __init__(self, length=100):
        self.state = []
        self.winner = []
        self.strategy = []
        self.length = length

    def get_length(self):
        return len(self.state)

    def push(self, state, strategy, value):
        self.state.append(state)
        strategy_len = np.zeros(15 * 15)
        strategy_len[strategy] = 1
        self.strategy.append(strategy_len)
        self.winner.append(value)
        if len(self.state) >= self.length:
            self.state = self.state[1:]
            self.winner = self.winner[1:]
            self.strategy = self.strategy[1:]

    def copy(self, temp, winner):
        for i in range(0, len(temp.state)):
            self.state.append(temp.state[i])
            self.strategy.append(temp.strategy[i])
            self.winner.append(winner)
            if len(self.state) >= self.length:
                self.state = self.state[1:]
                self.winner = self.winner[1:]
                self.strategy = self.strategy[1:]

    def seq(self):
        return self.state, self.strategy, self.winner


def get_dataloader(stack, Batch_size):
    state, strategy, winner = stack.seq()
    tensor_x = torch.stack(tuple([torch.tensor(s) for s in state]))
    tensor_y1 = torch.stack(tuple([torch.tensor(y1) for y1 in strategy]))
    tensor_y2 = torch.stack(tuple([torch.Tensor([y2]) for y2 in winner]))
    dataset = torch_data.TensorDataset(tensor_x, tensor_y1, tensor_y2)
    my_loader = torch_data.DataLoader(dataset, batch_size=Batch_size, shuffle=True)
    return my_loader


batch_size = 32
Epoch = 20

Net = nn(input_layers=4, board_size=15, learning_rate=0.001)


def change(loss, temp):
    for a in temp:
        loss.append(a)


def train():
    queue = random_queue()
    tree = MCTS(impact=2, net=Net.eval)
    loss = []
    for train_epoch in range(0, Epoch):
        print("the {} self playing".format(train_epoch))
        tree.return_root(train_epoch)
        state = Board()
        temp_queue = random_queue()
        is_end = 0
        while not is_end:
            action, pi = tree.simulate(state)
            # print(action)
            # print(pi)
            # if train_epoch <= 3:
            #    pso = PSO(len(action)+1, action, state, iter_num=50)
            #    _, pso_value = pso.update()
            #    pi = pi + pso_value
            action_p = action[np.argmax(pi)]
            temp_queue.push(state.current_state(), action_p, 0)
            state.do_move(action_p)
            s, n = state.board_state()
            print(s)
            print(n)
            is_end, winner = state.game_end()
            # print(is_end)
            if is_end:
                print("end, the winner is {}".format(winner))
                queue.copy(temp_queue, winner)
                length = queue.get_length()
                if length >= batch_size:
                    data_loader = get_dataloader(queue, batch_size)
                    temp_loss = Net.train(data_loader)
                    for a in temp_loss:
                        loss.append(a)
            tree.reuse(action_p)
    x = list(range(len(loss)))
    plt.plot(x, loss)
    plt.ylabel("loss")
    plt.show()


def chess(my_role=1):  # 玩家1先下
    tree = MCTS(impact=2, net=Net.eval)
    is_end = 0
    winner = 0
    print("Game begin, player 1 will be the first, you are the player:{}".format(my_role))
    if my_role == -1:
        state = Board(first_one=1)
        s, _ = state.board_state()
        print(s)
        action, value = tree.simulate(state)
        pso = PSO(len(action) + 1, action, state)
        _, pso_value = pso.update()
        Value = value + pso_value
        action_oppo = action[np.argmax(Value)]
        state.do_move(action_oppo)
        s, _ = state.board_state()
        print(s)
        tree.reuse(action_oppo)
    else:
        state = Board(first_one=0)
        s, _ = state.board_state()
        print(s)
    current_player = 1
    while not is_end:
        if current_player == 1:
            while True:
                h, w = map(int, input().split())
                action_my = state.loc2mov([h, w])
                if action_my in state.available_step:
                    break
                else:
                    print("error step")
            state.do_move(action_my)
            tree.root_node.add_child(action_my, 1)
            tree.reuse(action_my)
            s, _ = state.board_state()
            print(s)
            is_end, winner = state.game_end()
            current_player = 0
        else:
            action, value = tree.simulate(state)
            print(value)
            pso = PSO(len(action) + 1, action, state)
            _, pso_value = pso.update()
            print(pso_value)
            Value = value + pso_value
            action_oppo = action[np.argmax(Value)]
            state.do_move(action_oppo)
            tree.reuse(action_oppo)
            is_end, winner = state.game_end()
            s, _ = state.board_state()
            print(s)
            current_player = 1
    print("Game over, the winner is player {}".format(winner))


train()
chess()
