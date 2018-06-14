import pre
import matplotlib.pyplot as plt
import numpy as np
import random


class br_lr:
    def __init__(self, x, y, learning_rate=0.0000001, threshold=0.005):
        self.x = x
        self.y = y
        self.cnt = 0
        self.exit = 0
        self.len = x.shape[0]
        self.threshold = threshold
        self.rate = learning_rate
        #self.end = np.array([1, 1, 1])
        self.theta = np.ones((x.shape[1], 1))
        self.random_list = random.sample(range(self.len), self.len)
        pass

    def sigmoid(self, x):
        ret = np.longfloat(1 / (1 + np.longfloat(np.exp(-x))))
        return ret

    def compute_grad(self):
        tmp = self.sigmoid(
            np.dot(self.x[self.random_list[self.cnt]], self.theta)) - self.y[self.random_list[self.cnt]]
        self.grad = np.multiply(self.x[self.random_list[self.cnt]].T, tmp)
        pass

    def update(self):
        self.compute_grad()
        if abs(self.grad[0]) < self.threshold and abs(self.grad[1]) < self.threshold and abs(self.grad[2]) < self.threshold:
            print('-----------after %d train success-----------:' % (self.cnt))
            print('grad is  :', self.grad)
            print("theta is :", self.theta)
            #print('end is   :', self.end)
            return False

        #self.grad = np.multiply(self.grad, self.end)
        self.theta -= np.multiply(self.rate / 3, self.grad).reshape(3, 1)

        print('-----------after %d train-----------:' % (self.cnt))
        print('grad is  :', self.grad)
        print("theta is :", self.theta)
        #print('end is   :', self.end)
        print('\n')

        self.cnt += 1
        if self.cnt >= self.len:
            print('train data is not enough')
            return False

        return True

    def test(self, x):
        #self.theta[0][0] = 0.0000000002
        #self.theta[1][0] = -0.000000003
        #self.theta[2][0] = -0.000000003
        print('x1     :', x[1])
        print('dot x1 :', np.dot(x[1], self.theta))
        print('x60    :', x[200])
        print('dot x60:', np.dot(x[200], self.theta))
        self.test_y = [self.sigmoid(
            np.dot(x[cnt], self.theta)) - 0.5 for cnt in range(0, self.len)]
        print('test_y :', self.test_y[200])


pre_obj = pre.br_lr_pre(shape=(2, 300))
lr_obj = br_lr(pre_obj.x, pre_obj.y)

is_run = True
while is_run:
    is_run = lr_obj.update()

lr_obj.test(pre_obj.test)
pre_obj.test_draw(lr_obj.test_y, lr_obj.len)

'''
        if self.grad[0] < self.threshold:
            self.end[0] = 0
        if self.grad[1] < self.threshold:
            self.end[1] = 0
        if self.grad[2] < self.threshold:
            self.end[2] = 0
            '''

'''
while theta < 1:
    pre_obj.draw(theta)

'''
