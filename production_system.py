def judge(knowledge,dic,dic_flag):
    while True:
        flag=0
        for k in dic:
            if dic_flag[k]==0 and dic[k] not in knowledge and all(i in knowledge for i in k) :
                knowledge.append(dic[k])
                print(k,"-->",dic[k],knowledge)
                dic_flag[k]=1
                flag=1
        if flag==0:
            return knowledge

def back(knowledge,dic,dic_flag,result):
    print("反向推理")
    for i in result:
        dic[i]

def find_key(input_dict, value):
    output()
    for k, v in input_dict.items():
        if v == value:
            k
    return None


if __name__ == '__main__':
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
    print(r_dict,r_flag)

    print(len(r), len(r_flag))
    k=[17,20,19,2,9]
    knowledge=judge(k,r_dict,r_flag)
    print(knowledge)



