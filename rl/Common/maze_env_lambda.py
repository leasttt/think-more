import numpy as np
import tkinter as tk
import time
import pandas as pd

np.random.seed(2)

UNIT = 40  # pixels
MAZE_H = 6  # grid height
MAZE_W = 6  # grid width


class Maze(tk.Tk):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell
        place_list = [[2, 1], [1, 2], [1,3], [4, 1], [3, 4]]
        hell_center_list = map(lambda a: origin + np.array([UNIT * a[0], UNIT * a[1]]), place_list)
        hell_index_list = map(lambda a: self.canvas.create_rectangle(
            a[0] - 15, a[1] - 15,
            a[0] + 15, a[1] + 15,
            fill='black'), hell_center_list)
        self.hell_list = list(map(lambda a: self.canvas.coords(a), hell_index_list))

        # create oval
        oval_center = origin + UNIT * 3
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        wall = -1
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 'u' or action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
                wall = -1
            else:
                wall = 0
        elif action == 'd' or action == 1:  # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
                wall = -1
            else:
                wall = 1
        elif action == 'r' or action == 2:  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
                wall = -1
            else:
                wall = 2
        elif action == 'l' or action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
                wall = -1
            else:
                wall = 3

        # if wall > -1:
        #     self.wall_cross(wall)
        # else:
        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
        elif s_ in self.hell_list:
            reward = -1
            done = True
        elif wall > -1:
            reward = -1
            done = True
        else:
            reward = -0.1
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()

    def wall_cross(self, direction):
        list = [[0, -10], [0, 10], [10, 0], [-10, 0]]
        self.canvas.move(self.rect, list[direction][0], list[direction][1])
        self.canvas.coords(self.rect)
        self.render()
        time.sleep(0.5)
        self.canvas.move(self.rect, -list[direction][0], -list[direction][1])