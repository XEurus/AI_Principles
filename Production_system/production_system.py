def judge(knowledge,dic,dic_flag):
    """
    :param knowledge: 通过推理后的条件
    :param dic: 输入的字典，键为前件，值为后件
    :param dic_flag: 判断这个键值对是否被细分过的标志量
    :return: knowledge 最后一位为结果
    """
    while True:
        flag=0
        for k in dic: #遍历字典的键
            if dic_flag[k]==0 \
                    and dic[k] not in knowledge \
                    and all(i in knowledge for i in k) : #未被检测+不在已有之内+被包含
                knowledge.append(dic[k]) #通过细分增加条件
                print(k,"-->",dic[k],knowledge)
                dic_flag[k]=1
                flag=1
        if flag==0:
            return knowledge

def back_recursion(dic,known,data_base):# 递归对元素进行分解 分解后判断
    """
    :param dic: 输入的字典，键为前件，值为后件
    :param known: 已知的数据/条件
    :param data_base: 通过假设推理出来的条件
    :return: 1 匹配成功
    """
    for j in data_base:
        for i in dic:
            if dic[i]==j:
                data_base_copy=data_base.copy() #浅拷贝 不能改变原有列表 以便递归回来的遍历
                for k in i:
                    data_base_copy.append(k)
                data_base_copy.remove(j)
                print("字典从值寻到键",j,"-->",i)
                print("尝试匹配",data_base_copy,known)
                if all( m in known for m in data_base_copy):
                    #print("匹配成功")
                    return 1
                else:
                    result=back_recursion(dic, known, data_base_copy)
                    if result:
                        return result

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
    #反向推理数据初始化
    end_body=[22,23,24,25,29,30,14]

    #正向推理
    k=[17,20,19,2,9]
    knowledge=judge(k,r_dict,r_flag)
    if knowledge[-1] in end_body:
        print("结果是",knowledge[-1])
    else:
        print("无结果")

    #反向推理
    k = [17, 20, 19,9,2]
    for q in end_body:
        print("")
        print("假设是",q)
        result=back_recursion(r_dict,k,[q])
        if result:
            break
    if result:
        print("结果是",q)
    else:
        print("无结果")