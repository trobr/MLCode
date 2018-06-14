import pre_data as pd
import numpy as np


class br_lr:
    def __init__(self, x, y, learning_rate=0.001, threshold=1.2):
        self.x = x
        self.y = y
        self.rate = learning_rate
        self.grad = 0
        self.shape = (2, len(x) // 2)
        # init value of Theta (matrix 1 * 2)
        self.theta = np.array([[0, 0]])
        self.threshold = threshold  # np.array([[threshold] * 2])
        self.cnt = 0

    def sigmoid(self, x):
        ret = 1 / (1 + np.exp(-x))
        return ret

    def compute_grad(self):
        tmp = self.sigmoid(np.dot(self.theta, self.x) - self.y)
        self.grad = np.dot(tmp, self.x.T)

    def is_finish(self):
        pass

    def update(self):
        self.compute_grad()
        print(self.grad.shape)

        if self.grad[0][0] < self.threshold:
            print("after %d train. success!" % (self.cnt))
            print("grad  is :", self.grad)
            print("theta is :", self.theta)
            return False
        else:
            self.theta = self.theta - self.rate * self.grad / 2
            self.cnt = self.cnt + 1
            print("after %d train." % (self.cnt))
            print("grad is:", self.grad)
            print("theta is :", self.theta)
            print("\n")
        return True


pre = pd.br_pre()
x, y = pre.build_primitive_data()

lr_ins = br_lr(x, y, 5)

is_run = True
print("enter update")


while is_run:
    is_run = lr_ins.update()
