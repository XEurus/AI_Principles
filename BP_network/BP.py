import numpy as np
import torch
import time
import numpy
import  recode
import os
import torch.nn as nn

# ./MNIST\t10k-images-idx3-ubyte.gz
# ./MNIST\t10k-labels-idx1-ubyte.gz
# ./MNIST\train-images-idx3-ubyte.gz
# ./MNIST\train-labels-idx1-ubyte.gz

class network():
    def __init__(self):
        print("init")
        self.test_img,self.test_labels,self.train_img,self.train_labels=self.get_data()
        print("init_down")

    def get_data(self):
        test_img=recode.parse_mnist("./MNIST/t10k-images-idx3-ubyte.gz")
        test_labels=recode.parse_mnist("./MNIST/t10k-labels-idx1-ubyte.gz")
        train_img=recode.parse_mnist("./MNIST/train-images-idx3-ubyte.gz")
        train_labels=recode.parse_mnist("./MNIST/train-labels-idx1-ubyte.gz")
        return test_img,test_labels,train_img,train_labels

    def relu(self,a):
        return max(0,a)
    #def input(self):

    def convolution_np(self,img,kernal_shape=3):
        # kernal = torch.tensor([
        #     [1, 0, 1],
        #     [0, 1, 0],
        #     [1, 0, 1],
        # ])

        kernal = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
                        ])
        new_img_shape=(len(img[0])-(kernal_shape-1),len(img[1])-(kernal_shape-1))
        new_img=np.zeros(new_img_shape)
        for i in range(len(img[1])-(kernal_shape-1)): #列
            for k in range(len(img[0])-(kernal_shape-1)): #行
                matrix=img[k:k+3,i:i+3]
                con=np.multiply(kernal,matrix)
                mean=np.mean(con)
                new_img[k,i]=mean

        return new_img

    # def convolution_GPU(self,input_img):
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    #     input_tensor =torch.from_numpy(input_img)
    #     input_tensor = input_tensor.to(device)
    #
    #     conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)
    #     conv_layer = conv_layer.to(device)
    #
    #     # 查看卷积核的权重
    #     print("Initial kernel weights:")
    #     print(conv_layer.weight)
    #
    #     # 假设卷积核初始化为全1的卷积核（你可以根据需要设置卷积核的值）
    #     with torch.no_grad():
    #         conv_layer.weight = nn.Parameter(torch.ones_like(conv_layer.weight))
    #
    #     # 进行卷积操作
    #     output_tensor = conv_layer(input_tensor.unsqueeze(0))  # 添加一个batch维度
    #     output_tensor = output_tensor.squeeze(0)  # 去掉batch维度
    #
    def pooling(self,input_img):


    # # def input_layer_init(self,shape):



if __name__ == "__main__":
    net=network()
    a=net.relu(-1)
    print(a)
    net.convolution_np(net.train_img[0])
    #net.get_data()