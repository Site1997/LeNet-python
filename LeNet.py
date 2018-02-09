# -*- coding: utf-8 -*-
'''
Author: Site Li
Website: http://blog.csdn.net/site1997
'''
import numpy as np
from scipy.signal import convolve2d
from skimage.measure import block_reduce
import fetch_MNIST


class LeNet(object):
    #The network is like:
    #    conv1 -> pool1 -> conv2 -> pool2 -> fc1 -> relu -> fc2 -> relu -> softmax
    # l0      l1       l2       l3        l4     l5      l6     l7      l8        l9
    def __init__(self, lr=0.1):
        self.lr = lr
        # 6 convolution kernal, each has 5 * 5 size
        self.conv1 = 2 * np.random.random((6, 1, 5, 5)) - 1
        # the size for max_pool is 2 * 2, stride = 2
        self.pool1 = [2, 2]
        # 16 convolution kernal, each has 6 * 5 * 5 size
        self.conv2 = 2 * np.random.random((16, 6, 5, 5)) - 1
        # the size for max_pool is 2 * 2, stride = 2
        self.pool2 = [2, 2]
        # fully connected layer 256 -> 200
        self.fc1 = 2 * np.random.random((256, 200)) - 1
        # relu layer
        # fully connected layer 200 -> 10
        self.fc2 = 2 * np.random.random((200, 10)) - 1
        # relu layer
        # softmax layer

    def forward_prop(self, input_data):
        self.l0 = np.expand_dims(input_data, axis=1) / 255   # (batc_sz, 1, 28, 28)
        self.l1 = self.convolution(self.l0, self.conv1)      # (batc_sz, 6, 24, 24)
        self.l2 = self.mean_pool(self.l1, self.pool1)        # (batc_sz, 6, 12, 12)
        self.l3 = self.convolution(self.l2, self.conv2)      # (batc_sz, 16, 8, 8)
        self.l4 = self.mean_pool(self.l3, self.pool2)        # (batc_sz, 16, 4, 4)
        self.l5 = self.fully_connect(self.l4, self.fc1)      # (batch_sz, 200)
        self.l6 = self.relu(self.l5)                         # (batch_sz, 200)
        self.l7 = self.fully_connect(self.l6, self.fc2)      # (batch_sz, 10)
        self.l8 = self.relu(self.l7)                         # (batch_sz, 10)
        self.l9 = self.softmax(self.l8)                      # (batch_sz, 10)
        # TODO : need loss function ?
        return self.l9

    def backward_prop(self, softmax_output, output_label):
        # TODO : decide which delta to use
        #l8_delta             = np.sum(output_label - softmax_output, axis=0)/batch_sz       # (batch_sz, 10)
        #l8_delta             = softmax_output - output_label
        l8_delta             = output_label - softmax_output
        l7_delta             = self.relu(self.l8, l8_delta, deriv=True)                     # (batch_sz, 10)
        l6_delta, self.fc2   = self.fully_connect(self.l6, self.fc2, l7_delta, deriv=True)  # (batch_sz, 200)
        l5_delta             = self.relu(self.l6, l6_delta, deriv=True)                     # (batch_sz, 200)
        l4_delta, self.fc1   = self.fully_connect(self.l4, self.fc1, l5_delta, deriv=True)  # (batch_sz, 16, 4, 4)
        l3_delta             = self.mean_pool(self.l3, self.pool2, l4_delta, deriv=True)    # (batch_sz, 16, 8, 8)
        l2_delta, self.conv2 = self.convolution(self.l2, self.conv2, l3_delta, deriv=True)  # (batch_sz, 6, 12, 12)
        l1_delta             = self.mean_pool(self.l1, self.pool1, l2_delta, deriv=True)    # (batch_sz, 6, 24, 24)
        l0_delta, self.conv1 = self.convolution(self.l0, self.conv1, l1_delta, deriv=True)  # (batch_sz, 6, 12, 12)


    def convolution(self, input_map, kernal, front_delta=None, deriv=False):
        N, C, W, H = input_map.shape
        K_NUM, K_C, K_W, K_H = kernal.shape
        if deriv == False:
            feature_map = np.zeros((N, K_NUM, W-K_W+1, H-K_H+1))
            for imgId in range(N):
                for kId in range(K_NUM):
                    for cId in range(C):
                        # TODO multi kernals; kernal[kId,::-1,::-1]?;
                        feature_map[imgId][kId] += convolve2d(input_map[imgId][cId], kernal[kId,cId,:,:], mode='valid')
            return feature_map
        else :
            # front->back (propagate loss)
            back_delta = np.zeros((N, C, W, H))
            kernal_gradient = np.zeros((K_NUM, K_C, K_W, K_H))
            for imgId in range(N):
                for cId in range(C):
                    for kId in range(K_NUM):
                        padded_front_delta = \
                          np.pad(front_delta[imgId][kId], [(K_W-1, K_H-1),(K_W-1, K_H-1)], mode='constant', constant_values=0)
                        back_delta[imgId][cId] += \
                          convolve2d(padded_front_delta, kernal[kId,cId,::-1,::-1], mode='valid')
                        kernal_gradient[kId][cId] += \
                          convolve2d(front_delta[imgId][kId], input_map[imgId][cId][::-1][::-1], mode='valid')
            # update weights
            kernal += self.lr * kernal_gradient
            return back_delta, kernal

    def mean_pool(self, input_map, pool, front_delta=None, deriv=False):
        N, C, W, H = input_map.shape
        P_W, P_H = tuple(pool)
        if deriv == False:
            feature_map = np.zeros((N, C, W/P_W, H/P_H))
            feature_map[:][:] = block_reduce(input_map[:][:], tuple((1, 1, P_W, P_H)), func=np.mean)
            return feature_map
        else :
            # front->back (propagate loss)
            back_delta = np.zeros((N, C, W, H))
            back_delta = front_delta.repeat(P_W, axis = 2).repeat(P_H, axis = 3)
            back_delta /= (P_W * P_H)
            return back_delta

    def fully_connect(self, input_data, fc, front_delta=None, deriv=False):
        N = input_data.shape[0]
        if deriv == False:
            input_data = input_data.reshape(N, -1)
            output_data = np.dot(input_data, fc) # dtype=np.float64
            return output_data
        else :
            # front->back (propagate loss)
            back_delta = np.dot(front_delta, fc.T)
            back_delta = back_delta.reshape(input_data.shape)
            # update weights
            fc += self.lr * np.dot(input_data.reshape(N, -1).T, front_delta)
            return back_delta, fc

    def relu(self, x, front_delta=None, deriv=False):
        
        if deriv == False:
            #print x[0]
            return x * (x > 0)
        else :
            # propagate loss
            #print front_delta
            back_delta = front_delta * 1. * (x > 0)
            return back_delta

    def softmax(self, x):
        y = list()
        for t in x:
            e_t = np.exp(t - np.max(t))
            y.append(e_t / e_t.sum())
        return np.array(y)


def convertToOneHot(labels):
    oneHotLabels = np.zeros((labels.size, labels.max()+1))
    oneHotLabels[np.arange(labels.size), labels] = 1
    return oneHotLabels

if __name__ == '__main__':
    # size of data, batch size
    data_size = 10000; batch_sz = 64;
    # learning rate, max iteration
    lr = 0.00001;    max_iter = 5000;
    train_imgs = fetch_MNIST.load_test_images()
    train_labs = fetch_MNIST.load_test_labels().astype(int)
    train_labs = convertToOneHot(train_labs)
    #print np.max(train_imgs), np.min(train_imgs)
    my_CNN = LeNet(lr)
    for iters in range(max_iter):
        # starting index and ending index for input data
        st_idx = (iters % 155) * batch_sz;
        input_data = train_imgs[st_idx : st_idx + batch_sz]
        output_label = train_labs[st_idx : st_idx + batch_sz]
        softmax_output = my_CNN.forward_prop(input_data)
        if iters % 10 == 0:
            # calculate accuracy
            correct_list = [ int(np.argmax(softmax_output[i])==np.argmax(output_label[i])) for i in range(batch_sz) ]
            accuracy = float(np.array(correct_list).sum()) / batch_sz
            # calculate loss
            correct_prob = [ softmax_output[i][np.argmax(output_label[i])] for i in range(batch_sz) ]
            correct_prob = filter(lambda x: x > 0, correct_prob)
            loss = -1.0 * np.sum(np.log(correct_prob))
            print "-----------------------------------"
            print "The %d iters result:" % iters
            print correct_prob
            print "The accuracy is %f The loss is %f " % (accuracy, loss)
        my_CNN.backward_prop(softmax_output, output_label)
