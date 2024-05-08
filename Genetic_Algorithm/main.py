import random
import numpy as np
import matplotlib.pyplot as plt
import math


# def __init__(self):
#     self.city_number = 50  # 要求是偶数
#     # self.city_array=np.random.randint (0,self.city_number*10,size=[self.city_number,2])
#     self.city_array = np.random.randint(0, 100, size=[self.city_number, 2])
#     self.begin_index = 0
#     self.population = 500
#     self.path_array = np.empty((self.population, self.city_number + 1))
#     self.k = int(self.population / 100)  # self.k:锦标赛规模大小
#     self.keep = int(self.population / 2)  # 本代最后选取完的规模大小
#     self.top_times = 200
#     self.keep_excellent = int(self.population * 0.01)
#     self.p = 0
#     self.rate = 1
# if std_dev != 0:
#     self.p = 0.01 / (std_dev / mean)


class TSP():
    def __init__(self):
        self.city_number=20 #要求是偶数
        #self.city_array=np.random.randint (0,self.city_number*10,size=[self.city_number,2])
        self.city_array = np.random.randint(0, 100, size=[self.city_number, 2])
        self.begin_index=0
        self.population=500
        self.path_array=np.empty((self.population, self.city_number+1))
        self.k=int(self.population/4)             #self.k:锦标赛规模大小
        self.keep=int(self.population/3) #本代最后选取完的规模大小
        self.top_times=200
        self.keep_excellent=int(self.population*0.01)
        self.p=0
        self.rate=2

    def visualize_costs(self,costs, generation):
        plt.clf()
        plt.boxplot(costs)
        plt.title(f"Cost Distribution in Generation {generation}")
        plt.ylabel("Cost")
        plt.xlabel("Generation")
        plt.draw()
        plt.pause(0.1)

    def show(self, path=None,time=5,k=None):
        plt.clf()  # 清除当前图形
        # 绘制所有城市
        for i in range(len(self.city_array)):
            x = self.city_array[i, 0]
            y = self.city_array[i, 1]
            plt.scatter(x, y, color='red')
            plt.text(x, y, str(i), color='blue', fontsize=12)

        if path is not None:
            # 绘制路径
            for i in range(len(path) - 1):
                x = [self.city_array[path[i], 0], self.city_array[path[i + 1], 0]]
                y = [self.city_array[path[i], 1], self.city_array[path[i + 1], 1]]
                plt.plot(x, y, 'k-')
            # 返回到起点的线
            x = [self.city_array[path[-1], 0], self.city_array[path[0], 0]]
            y = [self.city_array[path[-1], 1], self.city_array[path[0], 1]]
            plt.plot(x, y, 'k-')

        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('TSP Path{}'.format(k))
        plt.draw()
        plt.pause(time)  # 暂停一段时间，例如0.1秒，让图更新显示出来



    def born_init(self):
        for i in range(self.population): #产生population个初始个体
            path=np.random.choice(range(1, self.city_number), self.city_number-1,replace=False)#每个个体从所有点中随机原则一条路，但是开始位置一样 大小是一个10行2列矩阵 产生population个矩阵
            path=np.insert(path,0, values=0) #插入开始位置
            fit=self.caculate_path_fit(path) #计算适应度/代价
            path=np.insert(path,self.city_number,values=fit) # 插入矩阵path最后一列
            #np.insert(self.path_array, 1, path, axis=0)
            self.path_array[i]=path #更新path_array


    def change(self):
        #arr=self.path_array
        n=int(self.p*self.population)
        rows_to_modify = np.random.choice(self.path_array.shape[0], n, replace=False)

        # 对每行执行随机选择索引和逆序操作
        for row_index in rows_to_modify:
            idx1, idx2 = np.sort(np.random.choice(self.path_array.shape[1]-2, 2, replace=False)+1)
            segment = self.path_array[row_index, idx1:idx2 + 1]
            reversed_segment = segment[::-1]
            self.path_array[row_index, idx1:idx2 + 1] = reversed_segment

    def reproduce(self):

        sorted_indices = np.argsort(self.path_array[:, -1]) #排名对应索引 将生成依照最末尾代价排序后的索引数组
        #sorted_indices=self.path_array[:,-1]
        #sorted_indices=sorted_indices.astype(int)
        excellent_arr=np.argsort(self.path_array[:, -1])[:self.keep_excellent]
        self.change()  # 对上代进行变异
        #print(sorted_indices)
        #sorted_indices_frozen=sorted_indices.copy()
        # 从索引中随机抽取 再根据索引产生下一次
        last_array=np.empty((1,self.keep)) #本代保存下来的数组索引列表  1行k列

        for i in range(self.keep-self.keep_excellent): #锦标赛获取本代优秀个体，每次获取一个，获取k次
            game_player =np.random.choice(sorted_indices, self.k, replace=False)#随机选择k规模 取出排名最靠前的
            member_win_index_in_game = np.argmin(self.path_array[game_player, -1])
            last_array[0, i] = game_player[member_win_index_in_game ]  # 本代保存下来的数组索引列表
            #win_path_index=np.where(np.isin(sorted_indices_frozen, member_win))[0] #排名的索引  在冻结索引数组中得到竞标赛最小排名的索引
            #with_path_index=sorted_indices_frozen[member_win]
            #print(win_path_index)
            delete_index=np.where(sorted_indices==game_player[member_win_index_in_game])[0]
            sorted_indices=np.delete(sorted_indices,delete_index )#锦标赛选择数组中删除该选手 按数字删除 不能按索引删除

        a=self.keep-self.keep_excellent
        for k in range(self.keep_excellent):
            last_array[0,k+a]=excellent_arr[k]
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

        k1=int((self.city_number-1)/4+1)
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
        path1_=np.insert(path1_,k1,k2)
        path1_=path1_.astype(int)
        fit = self.caculate_path_fit(path1_)  # 计算适应度/代价
        path1_ = np.insert(path1_, self.city_number, values=fit)  # 插入矩阵path最后一列
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
            _fit+=math.sqrt(fit)
        _fit+=math.sqrt((self.city_array[path[-1]][0]-self.city_array[path[0]][0])**2+(self.city_array[path[-1]][1]-self.city_array[path[0]][1])**2 )#最后一个点到出发点距离
        #_fit=math.sqrt(_fit)
        return _fit


    def main(self):
        self.born_init()
        plt.ion()  # 开启交互模式
        way = np.argsort(self.path_array[:, -1])
        way_index=way[0]
        best_weight=self.path_array[way_index,-1]
        best_path = self.path_array[way_index, :-1].astype(int)
        #print(best_weight)
        times=0
        for i in range(self.top_times):
            self.reproduce()

            min_index = np.argmin(self.path_array[:, -1])
            current_weight = self.path_array[min_index, -1]
            current_path = self.path_array[min_index, :-1].astype(int)
            # 打印当前代和当前最佳路径的信息
            #print(f"Generation {i + 1}: Best path weight = {current_weight}")
            costs=self.path_array[:, -1]
            #x=np.var(costs)
            mean = np.mean(costs)

            # 计算标准差
            std_dev = np.std(costs)

            #self.p = (1 - best_weight / mean)*0.5
            self.p = (1 - best_weight / mean)*self.rate
            #self.p =0.1
            #if std_dev/mean>0.01:
                #self.p=0.01/(std_dev/mean)
                #self.p=std_dev/mean
                #self.p=(mean/std_dev)*0.01
                #self.p = 0.01*(std_dev/ mean)
                #self.p=(1-best_weight/mean)
            #else:
            #    break
            print("std:",std_dev,"avg:",mean,"p:",self.p)
            #print(int(np.var(costs)))
            #self.show(current_path,0.1,i+1)  # 显示最佳路径
            self.visualize_costs(costs, i + 1)

            #plt.pause(0.01)  # 稍作暂停，以便观察变化
            # 检查是否有新的最优路径
            # if current_weight < best_weight:
            #     best_weight = current_weight
            #     best_path = self.path_array[min_index, :-1].astype(int)
            #     print(f"Generation {i + 1}: New best path found with weight {best_weight}")
            #     self.show(best_path)  # 绘制最佳路径

            #way = np.argsort(self.path_array[:, -1])
            #way_index = np.where(np.isin(way, 0))[0]
            #way_index=way[0]
            #temp = self.path_array[way_index, -1]
            print(f"Generation {i + 1}: best {best_weight},current {current_weight}")

            if best_weight==current_weight:
                times=times+1
            else:
                times=0
            print(times)
            if times==10:
                break
            if self.p<0.001:
               break
            if current_weight<best_weight:
                best_weight=current_weight
                best_path=current_path


            #print(temp)
        print("最优代价：",best_weight)
        print("最后一次收敛路径：",current_path)
        print("最优路径", best_path)
        self.show(current_path)
        self.show(best_path,-1)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tsp=TSP()
    tsp.main()

