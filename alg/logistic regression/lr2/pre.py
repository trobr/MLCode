import numpy as np
import matplotlib.pyplot as plt


class br_lr_pre:
    def __init__(self, shape=(3, 1000)):
        self.shape = shape
        self.dlen = shape[1] // 2

        y1 = [[0] * self.dlen]
        y2 = [[1] * self.dlen]
        y = np.concatenate((y1, y2))
        self.y = y.reshape(shape[1], 1)

        pri = np.random.uniform(1, 5, size=shape[1])
        x21 = [5 * i * i + np.random.uniform(1, 50)
               for i in pri[:self.dlen]]
        x22 = [5 * i * i - np.random.uniform(1, 50)
               for i in pri[self.dlen:]]

        x1 = pri
        x2 = np.concatenate((x21, x22))
        x = [[x1[i], x2[i], 1] for i in range(shape[1])]
        self.x = np.array((x))

        x1 = np.random.uniform(1, 5, size=shape[1])
        sort_pri = sorted(self.x[::, 1])
        x2 = np.random.uniform(
            sort_pri[0], sort_pri[self.dlen * 2 - 1], size=shape[1])
        x = [[x1[i], x2[i], 1] for i in range(shape[1])]
        self.test = np.array((x))

        plt.ioff()

    def draw(self, theta, delay=0.1):
        plt.clf()
        plt.scatter(self.x[:self.dlen, 0],
                    self.x[:self.dlen, 1], color='g', marker='o')
        plt.scatter(self.x[self.dlen:, 0],
                    self.x[self.dlen:, 1], color='b', marker='x')

    def save(self, path):
        with open(path, 'w+') as fd:
            for i in range(self.dlen):
                fd.write(str(self.x[i, 0]) +
                         ', ' + str(self.x[i, 1]) + ', ' + '1, 0' + '\n')
                fd.write(str(self.x[self.dlen + i, 0]) +
                         ', ' + str(self.x[self.dlen + 1, 1]) + ', ' + '0, 1' + '\n')

        #r_y = np.array(([np.dot(self.test_x.T, theta)]))
        # plt.pause(5)

    def test_draw(self, test_y, len):
        i = 0
        self.draw(0)
        while i < len:
            if test_y[i] < 0:
                plt.scatter(self.test[i][0], self.test[i]
                            [1], color='c', marker='^')
            else:
                plt.scatter(self.test[i][0], self.test[i]
                            [1], color='m', marker='v')
            i += 1
            if i % 1000 == 0:
                print('draw %d point' % (i))

        plt.pause(0)

    def hold_off(self):
        plt.ioff()

    def hold_on(self):
        plt.ion()

    def show(self):
        plt.show()


l = br_lr_pre()
l.save()
l.draw(1)

'''
class br_lr_pre:
    def __init__(self, shape=(2, 100)):
        self.shape = shape
        dlen = shape[1] // 2
        self.dlen = dlen

        y1 = [[0] * dlen]
        y2 = [[1] * dlen]
        y = np.concatenate((y1, y2))
        self.y = y.reshape(1, shape[1])

        pri = np.random.uniform(1, 5, size=shape[1])
        x21 = [78 * i + np.random.randint(1, shape[1]) for i in pri[:dlen]]
        x22 = [78 * i - np.random.randint(1, shape[1]) for i in pri[dlen:]]

        x1 = pri
        x2 = np.concatenate((x21, x22))
        x = [[x1[i], x2[i]] for i in range(shape[1])]
        self.x = np.array(x)

        self.fig = plt.figure()
        plt.ion()
        self.ax = self.fig.add_subplot(1, 1, 1)

    def pre_draw(self):
        plt.sca(self.ax)
        self.ax.scatter(self.x[:self.dlen, 0],
                        self.x[:self.dlen, 1], color='g', marker='o')
        self.ax.scatter(self.x[self.dlen:, 0],
                        self.x[self.dlen:, 1], color='b', marker='x')
        plt.show()

    def draw(self, theta):
        # self.ax = self.fig.add_subplot(1, 1, 1)
        plt.sca(self.ax)
        plt.clf()
        self.ax.plot(np.random.logistic(1, 100, 100))
        plt.pause(0.5)

    def hold_off(self):
        plt.sca(self.ax)
        plt.ioff()

    def hold_on(self):
        plt.sca(self.ax)
        plt.ion()

    def show(self):
        plt.sca(self.ax)
        plt.show()
'''
