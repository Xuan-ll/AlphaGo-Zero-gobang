import copy
import numpy as np


class Node(object):
    def __init__(self, parent, prior_p, is_root=0):
        self.parent = parent
        self.child = {}
        self.visit_counter = 0  # 总访问次数
        self.priorP = prior_p  # 先验概率，神经网络算出
        self.Q = 0
        self.U = 0
        self.is_root = is_root

    def add_child(self, action, prior_p):
        if action not in self.child:
            self.child[action] = Node(self, prior_p)

    def add_children(self, action_p, min_value=0):
        for i in range(0, len(action_p[0])):
            value = float(action_p[0][i])
            if value > min_value:
                self.add_child(i, value)

    def backup(self, leaf_value):
        self.visit_counter += 1
        # self.Q += 1.0 * (leaf_value - self.Q) / self.visit_counter
        self.Q = (self.Q*(self.visit_counter-1)+leaf_value)/self.visit_counter
        if self.is_root != 1:
            self.parent.backup(-leaf_value)

    def get_value(self, impact):
        self.U = impact * self.priorP * np.sqrt(self.parent.visit_counter) / (1.0 + self.visit_counter)
        # print(self.Q)
        # print(self.U)
        return self.Q + self.U

    def select(self, impact):
        return max(self.child.items(), key=lambda act_node: act_node[1].get_value(impact))

    def is_leaf(self):
        return self.child == {}

    def is_root_now(self):
        return self.is_root == 1


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class MCTS(object):
    def __init__(self, impact, net, epoch=50):
        self.root_node = Node(None, 1.0, is_root=1)
        self.impact = impact
        self.epoch_simulation = epoch
        self.Net = net
        self.root_copy = self.root_node

    def return_root(self, epoch):
        if epoch != 0:
            self.root_node.is_root = 0
            self.root_node = self.root_copy
            self.root_node.is_root = 1

    def search(self, state):  # 自上而下的一次搜索
        node = self.root_node
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self.impact)
            state.do_move(action)
        is_end, action_p, value = self.evaluate(state)
        if not is_end:
            action_p = action_p * state.moved_step
            node.add_children(action_p)
        node.backup(-float(value))

    def reuse(self, last_move):
        if last_move in self.root_node.child:
            self.root_node.is_root = 0
            self.root_node = self.root_node.child[last_move]
            self.root_node.is_root = 1
            # self.root_node.parent = None
        else:
            print("reuse error")
            self.root_node = Node(None, 1.0)

    def evaluate(self, state):
        action_p, value = self.Net(state.current_state())
        is_end, winner = state.game_end()
        if is_end:
            if winner == 0:  # tie
                value = 0.0
            else:
                value = 1.0 if winner == state.get_current_player() else -1.0
        return is_end, action_p, value

    def play(self, temp=1e-3):
        act_visits = [(act, node.visit_counter) for act, node in self.root_node.child.items()]
        action, visits = zip(*act_visits)
        pi = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        # pi = np.power(visits, 1/temp)
        # pi = pi / np.sum(pi * 1.0)
        return action, pi

    def simulate(self, state, temp=1e-3):
        for n in range(0, self.epoch_simulation):
            state_copy = copy.deepcopy(state)
            self.search(state_copy)
        return self.play(temp)
