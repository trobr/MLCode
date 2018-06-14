#
#  ┏┓	┏┓
# ┏┛┻━━━┛┻┓
# ┃	      ┃
# ┃  ━    ┃
# ┃┳┛  ┗┳ ┃
# ┃	      ┃
# ┃	  ┻   ┃
# ┃	      ┃
# ┗━┓   ┏━┛
#	┃   ┃	神兽保佑
#	┃   ┃	代码无BUG!
#	┃   ┗━━━┓
#	┃ 		┣┓
#	┃ 		┏┛
#	┗┓┓┏━┳┓┏┛
#	 ┃┫┫ ┃┫┫
#	 ┗┻┛ ┗┻┛
#


from tensorflow.examples.tutorials.mnist import input_data
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import re
import numpy as np


class bp_net:
    def __init__(self):
        pass

    def init_data(self, batch_size, *targs, threshold=0.001,  learning_rate=0.8, glob_step=50000, active_function='relu', out_function='sigmoid', loss_function='mse'):
        self.current_step = 0
        self.glob_step = glob_step
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_layer = len(targs)                           # 输出层 隐藏层 输出层
        self.weight = []                                      # 权重
        self.bias = []                                        # 偏置
        self.layer = []                                       # 每一层值，未激活
        self.layer_active = []                                # 每一层激活值
        self.sigma = []                                       # 反向传播中sigma
        self.grad = []                                        # 反向传播梯度
        self.active_function = active_function
        self.derivative_active = 'diff_' + active_function
        self.out_function = out_function
        self.derivative_out = 'diff_' + out_function
        self.loss_function = loss_function
        self.derivative_loss = 'diff_' + loss_function

        self.layer.append(np.random.random((batch_size, targs[0])))
        self.layer_active.append(np.random.random((batch_size, targs[0])))

        for i in range(self.num_layer - 1):
            r, c = targs[i: i + 2]

            tw = np.random.normal(0, 0.1, (r, c))
            # tb = np.random.normal(0, 0.1, (1, c))
            tb = np.zeros((1, c))
            tn = np.random.normal(0, 0.1, (batch_size, c))
            ta = np.random.normal(0, 0.1, (batch_size, c))
            ts = np.random.normal(0, 0.1, (batch_size, c))
            tg = np.random.normal(0, 0.1, (r, c))

            self.weight.append(tw)
            self.bias.append(tb)
            self.layer.append(tn)
            self.layer_active.append(ta)
            self.sigma.append(ts)
            self.grad.append(tg)

       # self.weight = np.array(weight)
       # print(self.weight)
       # self.bias = np.asarray(bias)
       # self.layer = np.asarray(layer)

    def relu(self, x):
        return (np.abs(x) + x) / 2

    def diff_relu(self, x):
        x[x >= 0] = 1
        x[x < 0] = 0
        return x

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def diff_sigmoid(self, x):
        s = self.sigmoid(x)
        return np.multiply(s, 1 - s)
        # return np.multiply(x, 1 - x)

    def normal(self, x):
        return x

    def diff_normal(self, x):
        return 1

    def softmax(self, x):
        orig_shape = x.shape
        if len(x.shape) > 1:
            # Matrix
            def exp_minmax(x): return np.exp(x - np.max(x))

            def denom(x): return 1.0 / np.sum(x)
            x = np.apply_along_axis(exp_minmax, 1, x)
            denominator = np.apply_along_axis(denom, 1, x)

            if len(denominator.shape) == 1:
                denominator = denominator.reshape((denominator.shape[0], 1))

            x = x * denominator
        else:
            # Vector
            x_max = np.max(x)
            x = x - x_max
            numerator = np.exp(x)
            denominator = 1.0 / np.sum(numerator)
            x = numerator.dot(denominator)

        assert x.shape == orig_shape
        return x

    def diff_softmax(self, x):
        tmp = np.multiply(-x, x)
        for i in range(len(tmp)):
            local = np.where(np.max(tmp[i]))
            tmp[i][local] = x[i][local] * (1 - x[i][local])

        return tmp

    def mse(self, x, y):
        return (y - x) ** 2 / 2

    def diff_mse(self, x, y):
        return (x - y)

    def coss_entropy(self, x, y):
        pass

    def diff_coss_entropy(self, x, y):
        pass

    def interface(self, in_data):
        self.layer[0] = in_data
        self.layer_active[0] = in_data

        for i in range(0, self.num_layer - 2):
            self.layer[i + 1] = np.matmul(
                self.layer_active[i], self.weight[i]) + self.bias[i]
            self.layer_active[i + 1] = getattr(self,
                                               self.active_function)(self.layer[i + 1])

        self.layer[self.num_layer - 1] = np.matmul(
            self.layer_active[self.num_layer - 2], self.weight[self.num_layer - 2]) +\
            self.bias[self.num_layer - 2]
        self.layer_active[self.num_layer - 1] = getattr(
            self, self.out_function)(self.layer[self.num_layer - 1])
        return self.layer_active[self.num_layer - 1]

    def compute_loss(self, labels):
        self.err = labels - self.layer_active[self.num_layer - 1]
        self.loss = (self.err ** 2) / 2

    def fit_break(self):
        if (self.loss < self.threshold).all() or self.current_step > self.glob_step:
            print('after %d train' % (self.current_step))
            print('loss is :')
            print(self.loss)
            return True
        else:
            return False

    def compute_grad(self):
        # 损失函数对yo的梯度
        tmp = -self.err
        # sigma = tmp .* (dyo/dyi)
        # self.diff_sigmoid(self.layer[self.num_layer - 1]))
        self.sigma[self.num_layer - 2] = np.multiply(
            tmp, getattr(self, self.derivative_out)(self.layer[self.num_layer - 1]))
        self.grad[self.num_layer - 2] = np.matmul(
            self.layer_active[self.num_layer - 2].T, self.sigma[self.num_layer - 2]) / self.batch_size

        for i in range(self.num_layer - 3, -1, -1):
            # self.sigma[i] = np.matmul(self.sigma[i + 1], self.weight[i + 1].T)
            self.sigma[i] = np.matmul(self.sigma[i + 1], self.weight[i + 1].T)
            self.sigma[i] = np.multiply(self.sigma[i], getattr(
                self, self.derivative_active)(self.layer[i + 1]))
            self.grad[i] = np.matmul(
                self.layer_active[i].T, self.sigma[i]) / self.batch_size

    def updata_weight(self):
        for i in range(self.num_layer - 1):
            tmp = np.multiply(self.learning_rate, self.grad[i])
            # print('tmp\n', tmp)
            self.weight[i] -= np.multiply(self.learning_rate,
                                          self.grad[i]) / self.batch_size
            # print('weight[%d]' % (i))
            # print(self.weight[i])

    def train(self, input_data, labels):
        self.labels = labels
        self.current_step += 1
        self.interface(input_data)
        self.compute_loss(labels)
        # self.debug_priv()
        if self.fit_break():
            return True
        else:
            self.compute_grad()
            self.updata_weight()
            # self.debug()
            return False

    def debug_priv(self):
        print('-' * 100)
        for i in range(self.num_layer - 1):
            print('layer[%d]' % (i))
            print(self.layer[i])
            print('layer_active[%d]' % (i))
            print(self.layer_active[i])
            print('weight[%d]' % (i))
            print(self.weight[i])

        print('layer[%d]' % (self.num_layer - 1))
        print(self.layer[self.num_layer - 1])
        print('layer_active[%d]' % (self.num_layer - 1))
        print(self.layer_active[self.num_layer - 1])

    def debug(self):
        print('*' * 100)
        print('after %d train' % (cnt))
        print('self.err', self.err)
        print('self.labels', self.labels)
        # print('self.loss', self.loss)
        # print('self.layer - 1', self.layer[self.num_layer - 2])
        # print('self.layer_active - 1', self.layer_active[self.num_layer - 2])
        # print('self.weight', self.weight[self.num_layer - 2])
        # print('self.layer', self.layer[self.num_layer - 1])
        # print('self.layer_active', self.layer_active[self.num_layer - 1])
        # print('self.sigma[self.num_layer - 2]:\n',
        #       self.sigma[self.num_layer - 2])
        # print('self.grad', self.grad[self.num_layer - 2])
        for i in range(self.num_layer - 2, -1, -1):
            print('sigma[%d]' % (i))
            print(self.sigma[i])
            print('grad[%d]' % (i))
            print(self.grad[i])
            print('weight[%d]' % (i))
            print(self.weight[i])

        print('-' * 100)

    def test(self, input_data, labels):
        out = self.interface(input_data)
        print("label\n", labels)
        print('out\n', out)


'''
if __name__ == '__main__':
    BATCH_SIZE = 1

    mnist = input_data.read_data_sets(
        r'D:\ImgPro\DL&ML\TensorFlow\dataset\MNIST', one_hot=True)

    bp = bp_net()
    bp.init_data(BATCH_SIZE, *(784, 500, 10))

    cnt = 0
    while True:
        xs, ys = mnist.train.next_batch(BATCH_SIZE)
        cnt += 1
        end = bp.train(xs, ys)
        if cnt % 1000 == 0:
            print(cnt)
            print(bp.loss)
        if end:
            break

    xs, ys = mnist.train.next_batch(10)
    bp.test(xs, ys)
'''


def get_batch(fd, batch_size):
    if fd == None:
        print('fd err')
        return False

    xs = []
    ys = []
    for i in range(batch_size):
        msg = fd.readline()
        if (msg == ''):
            fd.seek(0)
            msg = fd.readline()
        msg = re.split(r'[\s,\\n]+', msg)

        x1, x2, y1, y2, *_ = msg
        x1 = float(x1)
        x2 = float(x2)
        y1 = float(y1)
        y2 = float(y2)
        xs.append([x1, x2])
        ys.append([y1, y2])
        # ys.append([y1])

    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys


if __name__ == '__main__':
    BATCH_SIZE = 1
    fd = open(r'D:\ImgPro\DL&ML\ML\code\nonline.txt')
    test_fd = open(r'D:\ImgPro\DL&ML\ML\code\nonline_test.txt')

    bp = bp_net()
    bp.init_data(BATCH_SIZE, *(2, 5, 2))

    cnt = 0
    while True:
        if cnt == 923:
            print('s')
        xs, ys = get_batch(fd, BATCH_SIZE)
        cnt += 1
        end = bp.train(xs, ys)
        if end:
            break

    xs, ys = get_batch(test_fd, 10)
    bp.test(xs, ys)

    fd.close()
    test_fd.close()
