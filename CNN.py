import numpy as np
from scipy.signal import convolve2d
from skimage.measure import block_reduce
import fetch_MNIST

class CNN(object):
    #The network is like:
    #    conv1 -> pool1 -> conv2 -> pool2 -> fc1 -> relu -> fc2 -> relu -> softmax
    # l0      l1       l2       l3        l4     l5      l6     l7      l8        l9
    def __init__(self):
        # 32 * 32 for input
        # 6 convolution kernal, each has 5 * 5 size,       output: 6 * (24,24)
        self.conv1 = 2*np.random.random((6, 5, 5)) - 1                               # (6, 5, 5)
        # the size for max_pool is 2 * 2, stride = 2           output: 6 * (12, 12)
        self.pool1 = [2, 2]
        # 16 convolution kernal, each has 5 * 5 size output: 16 * (8, 8) 
        self.conv2 = 2*np.random.random((16, 5, 5)) - 1                               # (6, 5, 5)
        # the size for max_pool is 2 * 2, stride = 2             output: 16 * (4, 4)
        self.pool2 = [2, 2]
        # fully connected layer 256 -> 200               output: 200
        self.fc1 = 2*np.random.random((256, 200)) - 1                                # (400, 200)
        # relu layer
        # fully connected layer 200 -> 10                 output: 10
        self.fc2 = 2*np.random.random((200, 10)) - 1                                 # (200, 10)
        # relu layer
        # softmax layer
    def forward_prop(self, input_data):
        self.l0 = np.expand_dims(input_data, axis=1) / 255   # (batc_sz, 1, 28, 28)
        self.l1 = self.convolution(self.l0, self.conv1)      # (batc_sz, 6, 24, 24)
        self.l2 = self.max_pool(self.l1, self.pool1)         # (batc_sz, 6, 12, 12)
        self.l3 = self.convolution(self.l2, self.conv2)      # (batc_sz, 16, 8, 8)
        self.l4 = self.max_pool(self.l3, self.pool2)         # (batc_sz, 16, 4, 4)
        self.l5 = self.fully_connect(self.l4, self.fc1)      # (batch_sz, 200)
        self.l6 = self.relu(self.l5, deriv=False)            # (batch_sz, 200)
        self.l7 = self.fully_connect(self.l6, self.fc2)      # (batch_sz, 10)
        self.l8 = self.relu(self.l7, deriv=False)            # (batch_sz, 10)
        print self.l8
        self.l9 = self.softmax(self.l8)                      # (batch_sz, 10)
        return self.l9

    def backward_prop(self, softmax_output, output_label):
        l8_delta = softmax_output - output_label
        l7_delta = l8_delta * self.relu(self.l8, deriv=True)
        l6_delta = np.dot(l7_delta, self.fc2.T)
        l5_delta = l6_delta * self.relu(self.l6, deriv=True)
        l4_delta = np.dot(l5_delta, self.fc1.T)

        #TODO

        '''
        l7_delta = np.dot(l8_delta, self.w2.T) * self.sigmoid(self.l1, True)         # (batch_sz , 5)
        self.w2 += lr * np.array(np.dot(self.l1.T, l2_delta), dtype=np.float32)      # (5 , 1)
        self.w1 += lr * np.array(np.dot(self.l0.T, l1_delta), dtype=np.float32)      # (4 , 5)
        '''

    def convolution(self, input_map, kernal):
        N, C, W, H = input_map.shape
        K_NUM, K_W, K_H = kernal.shape
        feature_map = np.zeros((N, K_NUM, W-K_W+1, H-K_H+1))
        for imgId in range(N):
            for kId in range(K_NUM):
                for cId in range(C):
                    feature_map[imgId][kId] += convolve2d(input_map[imgId][cId], kernal[kId,::-1,::-1], mode='valid')
                feature_map[imgId][kId] /= C
        return feature_map

    def max_pool(self, input_map, pool):
        N, C, W, H = input_map.shape
        feature_map = np.zeros((N, C, W/pool[0], H/pool[1]))
        for imgId in range(N):
            for cId in range(C):
                feature_map[imgId][cId] = block_reduce(input_map[imgId][cId], tuple(pool), func=np.max)
        return feature_map

    def fully_connect(self, input_map, fc):
        N = input_map.shape[0]
        input_map = input_map.reshape(N, -1)
        output = np.array(np.dot(input_map, fc), dtype=np.float64)
        return output

    def relu(self, x, deriv=False):
        if deriv == True:
            return 1. * (x > 0)
        return x * (x > 0)

    def softmax(self, x):
        N, class_num = x.shape
        row_max = np.array([np.max(x[i])*np.ones(class_num) for i in range(N)])
        e_x = np.exp(x - row_max)
        row_exp_sum = np.array([np.sum(e_x[i])*np.ones(class_num) for i in range(N)])
        return e_x / row_exp_sum


def convertToOneHot(labels):
    oneHotLabels = np.zeros((labels.size, labels.max()+1))
    oneHotLabels[np.arange(labels.size), labels] = 1
    return oneHotLabels

if __name__ == '__main__':
    # size of data, batch size
    data_size = 60000; batch_sz = 10; # 64
    # learning rate, max iteration
    lr = 0.1;      max_iter = 5000;
    train_imgs = fetch_MNIST.load_test_images()
    train_labs = fetch_MNIST.load_test_labels().astype(int)
    train_labs = convertToOneHot(train_labs)
    print train_labs.shape
    #print np.max(train_imgs), np.min(train_imgs)
    my_CNN = CNN()
    for iters in range(max_iter):
        # starting index and ending index for input data
        st_idx = (iters % 937) * batch_sz;
        input_data = train_imgs[st_idx : st_idx + batch_sz]
        output_label = train_labs[st_idx : st_idx + batch_sz]
        softmax_output = my_CNN.forward_prop(input_data)
        my_CNN.backward_prop(softmax_output, output_label)
        break
        '''
        if iters % 50 == 0:
            correct_list = [int(np.argmax(softmax_output[i])==np.argmax(output_label[i])) for i in range(batch_sz)]
            accuracy = float(np.array(correct_list).sum()) / batch_sz
            print "The accuracy is %f" % (accuracy)
            continue
        '''
        #my_CNN.backward_prop(softmax_output, output_label)
