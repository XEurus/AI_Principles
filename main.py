import random
import numpy as np
import matplotlib.pyplot as plt

class TSP():
    def __init__(self):
        self.city_number=10 #要求是偶数
        #self.city_array=np.random.randint(0,self.city_number*10,size=[self.city_number,2])
        self.city_array = np.random.randint(0, 10, size=[self.city_number, 2])
        self.begin_index=0
        self.population=1000
        self.path_array=np.empty((self.population, self.city_number+1))
        self.k=500             #self.k:锦标赛规模大小
        self.keep=int(self.population/2) #本代最后选取完的规模大小
        self.top_times=1000
    def show(self):
        for i in range(len(self.city_array)):
            x=self.city_array[i,0]
            y=self.city_array[i,1]
            plt.scatter(x,y)
        #print(self.city_array)

        plt.show()

    def born_init(self):
        for i in range(self.population): #产生population个初始个体
            path=np.random.choice(range(1, self.city_number), 9,replace=False)#每个个体从所有点中随机原则一条路，但是开始位置一样 大小是一个10行2列矩阵 产生population个矩阵
            path=np.insert(path,0, values=0) #插入开始位置
            fit=self.caculate_path_fit(path) #计算适应度/代价
            path=np.insert(path,10,values=fit) # 插入矩阵path最后一列
            #np.insert(self.path_array, 1, path, axis=0)
            self.path_array[i]=path #更新path_array

    def reproduce(self):
        sorted_indices = np.argsort(self.path_array[:, -1]) #排名对应索引 将生成依照最末尾代价排序后的索引数组
        #print(sorted_indices)
        sorted_indices_frozen=sorted_indices.copy()
        # 从索引中随机抽取 再根据索引产生下一次
        last_array=np.empty((1,self.keep)) #本代保存下来的数组索引列表  1行k列

        for i in range(self.keep): #锦标赛获取本代优秀个体，每次获取一个，获取k次
            game_player =np.random.choice(sorted_indices, self.k, replace=False)#随机选择k规模 取出排名最靠前的
            member_win=min(game_player) #排名
            win_path_index=np.where(np.isin(sorted_indices_frozen, member_win))[0] #排名的索引  在冻结索引数组中得到竞标赛最小排名的索引
            #print(win_path_index)
            delete_index=np.where(np.isin(sorted_indices,member_win))[0]
            sorted_indices=np.delete(sorted_indices,delete_index )#锦标赛选择数组中删除该选手 按数字删除 不能按索引删除
            last_array[0,i]=win_path_index# 本代保存下来的数组索引列表
        #print(last_array)
        last_array=last_array[0] #降低维度为1维

        new_path=np.empty((self.population, self.city_number+1))
        k = 0
        for i in range(int(self.population*0.5)):#产生population/2次，每次产生两个下一代个体

            path_=np.random.choice(last_array, 2, replace=False) #选择两个准备交叉
            new_path[k]=self.crossover(path_[0], path_[1])
            new_path[k + 1] =self.crossover(path_[1], path_[0])
            k+=2
        #print(new_path)
        self.path_array=new_path


    def crossover(self,path0,path1):

        k1=int((self.city_number-1)/4)
        k2=self.path_array[int(path0)][k1:-k1-1] #-1是剔除最后一个 用于交换的基因片段
        '''
        从path0中选取片段，删除path1中对应数字，把这些数字放到中间，其他数字对半开前后方放，组成新path1
        path1对应处理，生成新path0
        '''
        #path0_=self.path_array[int(path0)].copy()
        path1_=self.path_array[int(path1)].copy()
        path1_ = np.delete(path1_, -1)#删除代价
        #print(path0_)
        for i in range(len(k2)):
            delete_index_path1_ = np.where(np.isin(path1_, k2[i]))[0]
            path1_=np.delete(path1_, delete_index_path1_)
        #print(path1_)
        #(int(len(path1_-1)/2))
        #print(path1_)
        path1_=np.insert(path1_,2,k2)
        path1_=path1_.astype(int)
        fit = self.caculate_path_fit(path1_)  # 计算适应度/代价
        path1_ = np.insert(path1_, 10, values=fit)  # 插入矩阵path最后一列
        #print(path1_)
        #print(k2)

        return path1_

    def caculate_path_fit(self,path):
        _fit=0
        for i in range(len(path)-1):
            #print(path)
            #print(path[i])
            #print(self.city_array[path[i+1]][1])
            fit=(self.city_array[path[i]][0]-self.city_array[path[i+1]][0])**2+(self.city_array[path[i]][1]-self.city_array[path[i+1]][1])**2 #走过的距离 欧几里得距离
            #print(fit)
            _fit+=fit
        _fit+=(self.city_array[path[-1]][0]-self.city_array[path[0]][0])**2+(self.city_array[path[-i]][1]-self.city_array[path[0]][1])**2 #最后一个点到出发点距离
        return _fit


    def main(self):
        self.born_init()
        way = np.argsort(self.path_array[:, -1])
        way_index = np.where(np.isin(way, 0))[0]
        weight=self.path_array[way_index,-1]
        print(weight)
        for i in range(self.top_times):
            self.reproduce()
            way = np.argsort(self.path_array[:, -1])
            way_index = np.where(np.isin(way, 0))[0]
            temp = self.path_array[way_index, -1]
            if temp<weight:
                weight=temp
            print(temp)
        print(weight)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tsp=TSP()
    tsp.main()

