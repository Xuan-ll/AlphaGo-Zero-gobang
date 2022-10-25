import numpy as np
import random


def mov2loc(move, width):
    h = move // width
    w = move % width
    return h, w


def fitness_value_function(par, action, state):
    value = 0
    sum_pos = 0
    moved = list(set(range(state.width * state.height)) - set(state.available_step))
    for p in par.pos:
        sum_pos += p
    for i in range(len(action)):
        v = 0
        par.pos[i] = par.pos[i]/sum_pos
        for m in moved:
            h1, w1 = mov2loc(action[i], state.width)
            h2, w2 = mov2loc(m, state.width)
            v += (h1 - h2) ** 2 + (w1 - w2) ** 2
        value += v * par.pos[i]
    return value


class Particle:
    def __init__(self, x_max, v_max, num, action, state):
        self.pos = [random.uniform(0, x_max) for i in range(num)]
        self.vel = [random.uniform(-v_max, v_max) for i in range(num)]
        self.best_pos = [0.0 for i in range(num)]
        self.fitness_value = fitness_value_function(self, action, state)

    def set_pos(self, i, value):
        self.pos[i] = value

    def get_pos(self):
        return self.pos

    def set_best_pos(self, i, value):
        self.best_pos[i] = value

    def get_best_pos(self):
        return self.best_pos

    def set_vel(self, i, value):
        self.vel[i] = value

    def get_vel(self):
        return self.vel

    def set_fitness_value(self, value):
        self.fitness_value = value

    def get_fitness_value(self):
        return self.fitness_value


class PSO:
    def __init__(self, size, action, state, iter_num=100, x_max=1, v_max=0.01, w=0.6, c1=2, c2=2):
        self.C1 = c1
        self.C2 = c2
        self.W = w
        self.state = state
        self.action = action
        self.dim = len(action)  # 粒子的维度，即action的长度
        self.size = size  # 粒子个数
        self.x_max = x_max
        self.max_vel = v_max
        self.iter_num_max = iter_num  # 迭代最大次数
        self.best_fitness_value = float('Inf')
        self.best_position = [0.0 for i in range(self.dim)]  # 种群最优位置
        self.fitness_val_list = []  # 每次迭代最优适应值
        self.Particle_list = [Particle(self.x_max, self.max_vel, self.dim, self.action, self.state) for i in
                              range(self.size)]

    def set_bestFitnessValue(self, value):
        self.best_fitness_value = value

    def get_bestFitnessValue(self):
        return self.best_fitness_value

    def set_bestPosition(self, i, value):
        self.best_position[i] = value

    def get_bestPosition(self):
        return self.best_position

        # 更新速度

    def update_vel(self, part):
        for i in range(self.dim):
            vel_value = self.W * part.get_vel()[i] + self.C1 * random.random() * (
                    part.get_best_pos()[i] - part.get_pos()[i]) \
                        + self.C2 * random.random() * (self.get_bestPosition()[i] - part.get_pos()[i])
            if vel_value > self.max_vel:
                vel_value = self.max_vel
            elif vel_value < -self.max_vel:
                vel_value = -self.max_vel
            part.set_vel(i, vel_value)

        # 更新位置

    def update_pos(self, part):
        for i in range(self.dim):
            pos_value = part.get_pos()[i] + part.get_vel()[i]
            if pos_value < 0:
                pos_value = 0
            part.set_pos(i, pos_value)
        value = fitness_value_function(part, self.action, self.state)
        if value < part.get_fitness_value():
            part.set_fitness_value(value)
            for i in range(self.dim):
                part.set_best_pos(i, part.get_pos()[i])
        if value < self.get_bestFitnessValue():
            self.set_bestFitnessValue(value)
            for i in range(self.dim):
                self.set_bestPosition(i, part.get_pos()[i])

    def update(self):
        for i in range(self.iter_num_max):
            for part in self.Particle_list:
                self.update_vel(part)  # 更新速度
                self.update_pos(part)  # 更新位置
            self.fitness_val_list.append(self.get_bestFitnessValue())  # 每次迭代完把当前的最优适应度存到列表
        return self.fitness_val_list, self.get_bestPosition()
