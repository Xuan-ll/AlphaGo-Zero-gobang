import numpy as np
import torch


class Board(object):
    def __init__(self, width=15, height=15, first_one=0):
        self.width = width
        self.height = height
        self.player = [1, -1]
        self.current_player = self.player[first_one]
        self.available_step = list(range(self.width * self.height))
        self.moved_step = torch.ones(self.width * self.height)
        self.states = {}  # {move,player}   //落子处
        self.last_move = -1

    # 0 1 2 3
    # 4 5 6 7
    # 8 9 1011

    def mov2loc(self, move):
        h = move // self.width
        w = move % self.width
        return [h, w]

    def loc2mov(self, loc):
        if len(loc) != 2:
            return -1
        move = loc[0] * self.width + loc[1]
        if loc[0] < 0 or loc[0] >= self.width or loc[1] < 0 or loc[1] >= self.height:
            return -1
        return move

    def current_state(self):
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][self.mov2loc(move_curr)] = 1.0
            square_state[1][self.mov2loc(move_oppo)] = 1.0
            square_state[2][self.last_move // self.width, self.last_move % self.height] = 1.0
        if self.current_player == 1:
            square_state[3][:, :] = 1.0
        return square_state

    def do_move(self, move):
        self.states[move] = self.current_player
        self.available_step.remove(move)
        self.moved_step[move] = 0
        self.current_player = -self.current_player
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        # states = self.states
        win_n = 5
        moved = list(set(range(width * height)) - set(self.available_step))
        if len(moved) < win_n + win_n - 1:
            return False, -1
        for m in moved:
            h = m // width
            w = m % width
            player = self.states[m]
            if w in range(width - win_n + 1) and len(set(self.states.get(i, 0) for i in range(m, m + win_n))) == 1:
                # 某一行满足五子
                return True, player

            elif h in range(height - win_n + 1) and len(
                    set(self.states.get(i, 0) for i in range(m, m + win_n * width, width))) == 1:
                # 某一列满足五子
                return True, player

            elif w in range(width - win_n + 1) and h in range(height - win_n + 1) and len(
                    set(self.states.get(i, 0) for i in range(m, m + win_n * (width + 1), width + 1))) == 1:
                # 右对角线上满足五子
                return True, player

            elif w in range(win_n - 1, width) and h in range(height - win_n + 1) and len(
                    set(self.states.get(i, 0) for i in range(m, m + win_n * (width - 1), width - 1))) == 1:  # 左对角线
                return True, player
        return False, 0

    def game_end(self):
        win, winner = self.has_a_winner()
        if win:
            return True, winner  # 有人赢
        elif len(self.available_step) == 0:  # 平局
            return True, 0
        return False, 0

    def get_current_player(self):
        return self.current_player

    def board_state(self):
        square_state = np.zeros((self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == 1]
            move_oppo = moves[players != 1]
            square_state[self.mov2loc(move_curr)] = 1
            square_state[self.mov2loc(move_oppo)] = -1
        return square_state, len(self.available_step)

