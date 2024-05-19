import torch
import time
import numpy
import  recode
import os


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

    #def input(self):

if __name__ == "__main__":
    net=network()
    #net.get_data()