def judge(knowledge,dic,dic_flag):
    while True:
        flag=0
        for k in dic:
            if dic_flag[k]==0 and dic[k] not in knowledge and all(i in knowledge for i in k) : #未被检测+不在已有之内+被包含
                knowledge.append(dic[k])
                print(k,"-->",dic[k],knowledge)
                dic_flag[k]=1
                flag=1
        if flag==0:
            return knowledge

def back_recursion(dic,known,data_base):# 递归对元素进行分解 分解后判断
    for j in data_base:
        for i in dic:

            if dic[i]==j:
                data_base_copy=data_base.copy()
                for k in i:
                    data_base_copy.append(k)
                data_base_copy.remove(j)
                print("字典从值寻到键",j,"-->",i)
                #print(data_base_copy)
                print("尝试匹配",data_base_copy,known)
                if all( m in known for m in data_base_copy):
                    print("匹配成功")
                    return 1
                else:
                    result=back_recursion(dic, known, data_base_copy)
                    if result:
                        return result

"""
def back(knowledge,result,end_body):
    print("反向推理")
    tmp=[]
    for i in end_body:
        print("提出假设",i)
        tmp=[i]

        div_flag=1
        while div_flag == 1:
            div_flag=0
            for k in list(tmp):
                print(tmp)
                if not all(j in knowledge for j in tmp): # 如果已知未能包含全部已有的假设项
                    print("假设不成立，尝试获取假设的细分")
                    if k in result: # 进一步细分假设项
                        for l in result[k]:
                            tmp.append(l) #增加k的细分，删除k
                        tmp.remove(k)
                        div_flag=1
                        #k=0
                    else:print("%d无法细分",k)
                else:
                    print("假设成立",i)
"""

if __name__ == '__main__':
    #数据初始化
    a = ["1毛发", "2奶", "3羽毛", "4会飞", "5吃肉", "6犬齿", "7有爪", "8眼盯前方", "9有蹄",
         "10反刍动物", "11哺乳动物", "12鸟", "13善飞", "14信天翁", "15食肉动物", "16黄褐色", "17暗斑点",
         "18黑色条纹", "19长腿", "20长脖子", "21有蹄类动物", "22虎", "23金钱豹", "24长颈鹿", "25斑马",
         "26不会飞", "27黑白二色", "28会游泳", "29鸵鸟", "30企鹅", "31下蛋"]
    r = [[1, 11], [2, 11], [3, 12], [4, 31, 12], [5, 15], [6, 7, 8, 15], [11, 9, 21], [11, 10, 21],
         [11, 15, 16, 17, 23], [11, 15, 16, 18, 22], [21, 20, 19, 17, 24],
         [21, 18, 25], [12, 20, 26, 27, 24, 29], [12, 28, 16, 27, 30], [12, 13, 14]]
    r_dict = {}
    r_flag={}
    for item in r:
        key = tuple(item[:-1])  # 将除最后一位的部分作为键
        value = item[-1]  # 将最后一位作为值
        r_dict[key] = value
        r_flag[key] = 0
    #print(r_dict,r_flag)
    #print(len(r), len(r_flag))

    #反向推理数据初始化
    end_body=[22,23,24,25,29,30,14]
    back_dict={}
    back_flag={}
    for i in r:
        key = i[-1]
        value= tuple(i[:-1])
        back_dict[key]=value
        back_flag[key]=0
    print(back_dict,back_flag)



    #正向推理
    k=[17,20,19,2,9]
    #knowledge=judge(k,r_dict,r_flag)
    #print(knowledge)

    k = [17, 20, 19, 2, 9]
    #back(k,back_dict,end_body)
    #data_base=[22]
    for q in end_body:
        print("")
        print("假设是",q)
        result=back_recursion(r_dict,k,[q])
        if result:
            print("结果是",q)
            break

