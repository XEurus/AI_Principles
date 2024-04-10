import copy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pygraphviz as pgv
from networkx.drawing.nx_agraph import graphviz_layout

def calculate_w(matrix,end_matrix):
    _w=0
    for i in range(len(matrix)):
        for k in range(len(matrix[i])):
            if matrix[i][k]==end_matrix[i][k]:
                _w=_w+1
    w=9-_w
    return w

def find_next_state(matrix):
    index=[]
    for i in range(len(matrix)):
        for k in range(len(matrix[i])):
            if matrix[i][k]==0:
                index=[i,k]
                break
        if index!=[]:
            break
    print("position of zero:",index)

    m=index[0]
    n=index[1]
    x=len(matrix[0])-1

    matrix_list=[] # 维护一个矩阵列表

    #低维度矩阵直接写
    if n+1<=x:
        new_matrix=change_position([m, n],[m ,n + 1],matrix)
        matrix_list.append(new_matrix)
    if n-1>=0:
        new_matrix = change_position([m, n], [m, n - 1], matrix)
        matrix_list.append(new_matrix)
    if m+1<=x:
        new_matrix = change_position([m, n], [m + 1, n], matrix)
        matrix_list.append(new_matrix)
    if m-1>=0:
        new_matrix = change_position([m, n], [m - 1, n], matrix)
        matrix_list.append(new_matrix)

    return matrix_list

def change_position(index0,index1,matrix):
    matrix = copy.deepcopy(matrix)
    tmp=matrix[index0[0]][index0[1]]
    tmp2=matrix[index1[0]][index1[1]]
    matrix[index0[0]][index0[1]]=matrix[index1[0]][index1[1]]
    matrix[index1[0]][index1[1]]=tmp #原来的0位置赋予新值，新的位置直接赋0
    return matrix
def show(parent_matrix,sun_matrix,open_dict,G,pos):
    parent_matrix = copy.deepcopy(parent_matrix)
    sun_matrix = copy.deepcopy(sun_matrix)
    parent_matrix=np.array(parent_matrix)
    print(parent_matrix)
    for i in range(len(sun_matrix)):
        #G.add_edge(str(parent_matrix,open_dict[tuple(map(tuple,parent_matrix))]),str(np.array(sun_matrix[i]),open_dict[tuple(map(tuple, sun_matrix[i]))]))
        G.add_edge(str(parent_matrix) + ''.join(map(str, open_dict[tuple(map(tuple, parent_matrix))])),
                   str(np.array(sun_matrix[i])) + ''.join(map(str, open_dict[tuple(map(tuple, sun_matrix[i]))])))

        pos[str(np.array(sun_matrix[i]))] = (pos[str(parent_matrix)][0] - i/4, pos[str(parent_matrix)][1] - 3)
        print("----------")
        print(np.array(sun_matrix[i]))

    print("--------------------------------------")

if __name__ == '__main__':
    G = nx.DiGraph()
    end_matrix = [
        [1, 2, 3],
        [8, 0, 4],
        [7, 6, 5]
    ]

    init_matrix = [
        [2, 8, 0],
        [1, 4, 3],
        [7, 6, 5]
    ]

    open_dict={}
    close_dict={}
    h_start=calculate_w(init_matrix,end_matrix) #计算最优路径代价
    #print(tuple(init_matrix))
    open_dict[tuple(map(tuple, init_matrix))]=[h_start,0] # 初始化open表
    #path_list=[init_matrix]
    flag=0
    pos = {str(np.array(init_matrix)): (0, 0)}
    pos[str(np.array(init_matrix))+''.join(map(str,[6,0]))]= (0, 0)
    while True: #使用累加计算步长，矩阵作为键，代价作为值，代价=d+w，w由calculate函数计算
        next_state_=[]
        min_matrix_cost=float('inf')
        for key in open_dict.keys(): #找到最小的键，即代价最小的，对其进行扩展
            if min_matrix_cost>=open_dict[key][0]+open_dict[key][1]:
                min_matrix_cost=open_dict[key][0]+open_dict[key][1]
                min_matrix=key
        #path_list.append(list(map(list, min_matrix)))
        next_state=find_next_state(list(map(list, min_matrix))) #拓展代价最小的矩阵
        close_dict[min_matrix] = open_dict[min_matrix]  # 维护open和close两个表


        for i in range(len(next_state)): #检查新拓展出来的矩阵 计算其代价加入open表 如果到达就退出
            if tuple(map(tuple,next_state[i])) not in set(close_dict.keys()): #不走回头路
                next_state_.append(next_state[i])
                w=calculate_w(next_state[i],end_matrix)
                open_dict[tuple(map(tuple,next_state[i]))]=[w,open_dict[min_matrix][1]+1]
                if w==0:
                    print("find result:",next_state[i],open_dict[tuple(map(tuple, next_state[i]))])
                    flag=1
                    break

        show(list(map(list, min_matrix)), next_state_,open_dict, G, pos)
        del open_dict[min_matrix]
        if flag==1:
            break

    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.show()