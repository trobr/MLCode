import numpy as np
import matplotlib.pyplot as plt


class br_pre:
    def __init__(self):
        pass

    def build_primitive_data(self, dlen=100):
        y1 = [[0] * dlen]
        y2 = [[1] * dlen]
        y = np.concatenate((y1, y2))
        y = y.reshape(1, dlen * 2)

        pri = np.random.uniform(1, 5, size=(dlen * 2))
        # pri.sort()
        x21 = [3 * i + np.random.randint(1, dlen)
               for i in pri[:dlen]]
        x22 = [3 * i - np.random.randint(1, dlen)
               for i in pri[dlen:]]

        x1 = pri
        x2 = np.concatenate((x21, x22))

        x = np.concatenate((x1, x2))
        x = x.reshape(2, dlen * 2)
        return (x, y)

    def draw(self, x, y):
        dlen = len(x[0]) // 2
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(1, 1, 1)
        self.ax1.scatter(x[0][:dlen], x[1][:dlen],
                         marker='x', color='r', label='1')
        self.ax1.scatter(x[0][dlen:], x[1][dlen:],
                         marker='o', color='g', label='2')
        plt.show()
