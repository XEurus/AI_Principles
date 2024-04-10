#matrix=[[]*n for i in range(n)]#初始化一个n*n的零阵
import copy
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



if __name__ == '__main__':
    end_matrix = [
        [1, 2, 3],
        [8, 0, 4],
        [7, 6, 5]
    ]

    init_matrix = [
        [2, 8, 3],
        [1, 6, 4],
        [7, 0, 5]
    ]

    open_dict={}
    close_dict={}
    h_start=calculate_w(init_matrix,end_matrix) #计算最优路径代价
    #print(tuple(init_matrix))
    open_dict[tuple(map(tuple, init_matrix))]=[h_start,0] # 初始化open表
    path_list=[]
    flag=0
    while True: #使用累加计算步长，矩阵作为键，代价作为值，代价=d+w，w由calculate函数计算
        min_matrix_cost=float('inf')
        for key in open_dict.keys():
            if min_matrix_cost>=open_dict[key][0]+open_dict[key][1]:
                min_matrix_cost=open_dict[key][0]+open_dict[key][1]
                min_matrix=key
        next_state=find_next_state(list(map(list, min_matrix)))
        for i in range(len(next_state)):
            w=calculate_w(next_state[i],end_matrix)
            open_dict[tuple(map(tuple,next_state[i]))]=[w,open_dict[min_matrix][1]+1]
            if w==0:
                print("find result:",next_state[i],open_dict[tuple(map(tuple, next_state[i]))])
                flag=1
                break
        if flag==1:
            break
        close_dict[min_matrix]=open_dict[min_matrix]
        del open_dict[min_matrix]

        #next_state=find_next_state(init_matrix)
        #print(calculate_w(init_matrix,end_matrix))