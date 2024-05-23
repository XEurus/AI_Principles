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
        self.img_shape=np.zeros(20)
        self.fully_connection_layer1=None
        self.bias=None
        self.epoch=100
        self.batchsize=64
        self.cls=10
        print("init_down")

    def get_data(self):
        test_img=recode.parse_mnist("./MNIST/t10k-images-idx3-ubyte.gz")
        test_labels=recode.parse_mnist("./MNIST/t10k-labels-idx1-ubyte.gz")
        train_img=recode.parse_mnist("./MNIST/train-images-idx3-ubyte.gz")
        train_labels=recode.parse_mnist("./MNIST/train-labels-idx1-ubyte.gz")

        shuffled_indices = np.random.permutation(len(test_labels))
        # 按照相同的顺序打乱两个数组
        train_img = train_img[shuffled_indices]
        train_labels = train_labels[shuffled_indices]

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
        if len(input_img[0])%2==0 and len(input_img[1])%2==0: #要求偶数
            new_img_shape = (int(len(input_img[0])/2), int(len(input_img[1])/2))
            new_img = np.zeros(new_img_shape)
            for i in range(0,len(input_img[0]),2): #遍历行列
                for k in range(0,len(input_img[1]),2):
                    mean=np.mean(input_img[i:i+2,k:k+2]) #平均池化
                    new_img[int(i/2),int(k/2)]=mean
        else: #舍弃边缘一条的数据
            #print("池化层要求偶数矩阵输入")
            input_img=input_img[1:len(input_img[0]),1:len(input_img[1])]

            new_img_shape = (int(len(input_img[0]) / 2), int(len(input_img[1]) / 2))
            new_img = np.zeros(new_img_shape)
            for i in range(0, len(input_img[0]), 2):  # 遍历行列
                for k in range(0, len(input_img[1]), 2):
                    mean = np.mean(input_img[i:i + 2, k:k + 2])  # 平均池化
                    new_img[int(i / 2), int(k / 2)] = mean

        return new_img
    def create_weights(self,this_layer,next_layer):
        row=len(this_layer) #前一个输入提供行，后一个输出提供列
        """
                 [1,2  
        [1,2,3] @ 3,4  = [x1,x2]
                  5,6]
        """
        col=len(next_layer)
        return np.random.rand(row,col)

    def create_bias(self,shape):
        return np.random.randint(low=1,high=100,size=(shape))

    def fully_connection(self,input_matrix,input_weight,input_bias):
        output_matrix=np.dot(input_matrix,input_weight)+input_bias
        for i in range(len(output_matrix)):
            output_matrix[i]=self.relu(output_matrix[i])# 激活函数

        return output_matrix

    def softmax(self,input_matrix):
        max_ = np.max(input_matrix) #防止溢出
        softmax_out = np.exp(input_matrix-max_) / np.sum(np.exp(input_matrix-max_))
        for i in range(len(softmax_out)):
            if softmax_out[i]<1e-15:
                softmax_out[i]=1e-15
        return softmax_out

    def loss(self,vec1,vec2): #配合softmax使用交叉熵 vec1 vec2模型预测
        loss_=-np.sum(vec1*np.log(vec2))
        return loss_

    def one_hot(self,lable):
        lable_matrix=np.zeros(self.cls)
        lable_matrix[lable]=1
        return lable_matrix

    #def loss2softmax(self,y,   .):

    #def bp_w1(self,w

    def softmax_devide(self,o_matrix,index):
        sum_exp=np.sum(np.exp(o_matrix))
        Denominator=sum_exp*sum_exp
        exp_o_index=np.exp(o_matrix[index])
        numerator=exp_o_index*sum_exp-exp_o_index*exp_o_index

        return numerator/Denominator

    def caculate_divide(self,weight):


if __name__ == "__main__":
    net=network()

    index=0

    lable=net.train_labels[0]
    lable=net.one_hot(lable)

    cov_img=net.convolution_np(net.train_img[0])
    pool_img=net.pooling(cov_img)
    cov_img2=net.convolution_np(pool_img)
    pool_img2=net.pooling(cov_img2)

    input_array = pool_img2.reshape(-1) #拉长变成行向量

    hiden_layer = np.zeros(len(input_array))
    weight1=net.create_weights(input_array,hiden_layer)
    bias1=net.create_bias(len(input_array))

    output_layer=np.zeros(10)
    weight_hiden_output=net.create_weights(hiden_layer,output_layer)
    bias_hiden_output=net.create_bias(len(output_layer))

    hiden_layer=net.fully_connection(input_array,weight1,bias1)
    output_layer=net.fully_connection(hiden_layer,weight_hiden_output,bias_hiden_output)
    softmax_out=net.softmax(output_layer)
    loss1=net.loss(lable,softmax_out)

    weight_list=[weight1,weight_hiden_output]
    for i in range()


    for i in range(net.epoch):
        for k in range(net.batchsize):
            print("end")


    #net.get_data()

  
  
















