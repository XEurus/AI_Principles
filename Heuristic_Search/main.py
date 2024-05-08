import copy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def calculate_w2(matrix,end_matrix): #代价计算 计算不在位矩阵个数
    _w=0
    for i in range(len(matrix)):
        for k in range(len(matrix[i])):
            if matrix[i][k]==end_matrix[i][k] and matrix[i][k]!=0:
                _w=_w+1
    w=8-_w
    return w

def get_dis(matrix):
    dic={}
    for i in range(len(matrix)):
        for k in range(len(matrix[i])):
            dic[matrix[i][k]]=[i,k]
    return dic

def calculate_w(matrix,end_matrix):
    global end_dic
    dic=end_dic
    w=0
    for i in range(len(matrix)):
        for k in range(len(matrix[i])):
            num=matrix[i][k]
            dis_x=i-dic[num][0]
            dis_y=k-dic[num][1]
            m_dis=dis_x*dis_x+dis_y*dis_y
            w=w+m_dis
            #print(m_dis)
    #print(w)
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
    print("position of zero:",index) #找到0的位置

    m=index[0]
    n=index[1]
    x=len(matrix[0])-1

    matrix_list=[] # 维护一个矩阵列表

    #低维度矩阵直接写 往四个方向找
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

    return matrix_list #返回找到的矩阵

def change_position(index0,index1,matrix): #交换位置
    matrix = copy.deepcopy(matrix)
    tmp=matrix[index0[0]][index0[1]]
    matrix[index0[0]][index0[1]]=matrix[index1[0]][index1[1]]
    matrix[index1[0]][index1[1]]=tmp #原来的0位置赋予新值，新的位置直接赋0
    return matrix

def show(parent_matrix,son_matrix,open_dict,G,pos):
    parent_matrix = copy.deepcopy(parent_matrix)
    son_matrix = copy.deepcopy(son_matrix)
    parent_matrix=np.array(parent_matrix)

    p1 = d2s(parent_matrix, open_dict)
    print(p1)
    for i in range(len(son_matrix)):

        p2=d2s_son(son_matrix[i],open_dict)
        G.add_edge(p1,p2)
        pos[p2]=(pos[p1][0]-i/4,pos[p1][1] - 3)

def d2s(matrix,dict):
    s=(str(matrix) + ''.join(map(str, dict[tuple(map(tuple, matrix))])))
    return s
def d2s_son(son_matrix,open_dict):
    s=(str(np.array(son_matrix)) + ''.join(map(str, open_dict[tuple(map(tuple, son_matrix))])))
    return s


if __name__ == '__main__':

    G = nx.DiGraph()
    end_matrix = [
        [1, 2, 3],
        [8, 0, 4],
        [7, 6, 5]
    ]
    end_dic = get_dis(end_matrix)

    init_matrix = [
        [2, 8, 3],
        [1, 6, 4],
        [7, 0, 5]
    ]

    open_dict={}
    close_dict={}

    init_w=calculate_w(init_matrix,end_matrix)
    #print(tuple(init_matrix))
    open_dict[tuple(map(tuple, init_matrix))]=[init_w,0] # 初始化open表
    #path_list=[init_matrix]
    flag=0
    #pos = {str(np.array(init_matrix)): (0, 0)}
    pos={}
    pos[d2s_son(init_matrix,open_dict)]=(0,0)
    #pos[str(np.array(init_matrix))+''.join(map(str,[,0]))]= (0, 0)
    while True: #使用累加计算步长，矩阵作为键，代价作为值，代价=d+w，w由calculate函数计算
        next_state_=[]
        min_matrix_cost=float('inf')
        for key in open_dict.keys(): #找到最小的键，即代价最小的，对其进行扩展
            if min_matrix_cost>=open_dict[key][0]+open_dict[key][1]:
                min_matrix_cost=open_dict[key][0]+open_dict[key][1]
                min_matrix=key
        #path_list.append(list(map(list, min_matrix)))

        next_state=find_next_state(list(map(list, min_matrix))) #拓展代价最小的矩阵
        close_dict[min_matrix] = open_dict[min_matrix]  # 父节点放入close表

        for i in range(len(next_state)): #检查新拓展出来的矩阵 计算其代价加入open表 如果到达就退出
            if tuple(map(tuple,next_state[i])) not in set(close_dict.keys()): #不走回头路
                next_state_.append(next_state[i]) #可行的子节点列表
                w=calculate_w(next_state[i],end_matrix) #计算w
                open_dict[tuple(map(tuple,next_state[i]))]=[w,open_dict[min_matrix][1]+1] #b使用父节点累加计算
                if w==0:
                    print("find result:",next_state[i],open_dict[tuple(map(tuple, next_state[i]))])
                    flag=1
                    break

        show(list(map(list, min_matrix)), next_state_,open_dict, G, pos) #展示中添加父节点与子节点
        del open_dict[min_matrix] #删除open表父节点
        print("open-list：",open_dict)
        print("close-list:",close_dict)
        print("----------------------------------------")

        if flag==1:
            break

    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.show() #展示