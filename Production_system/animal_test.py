
# 特征数组
characteristic = ["毛发","奶","羽毛","会飞","吃肉","犬齿","有爪","眼盯前方","有蹄",
                  "反刍动物","哺乳动物","鸟","善飞","信天翁","食肉动物","黄褐色","暗斑点",
                  "黑色条纹","长腿","长脖子","有蹄类动物","虎","金钱豹","长颈鹿","斑马",
                  "不会飞","黑白二色","会游泳","鸵鸟","企鹅","下蛋"]

# 描述数组
describe = []

def BF(s, t):
    i = 0
    j = 0
    k = 0
    flag = -1
    while (i < len(s) and j < len(t)):

        # 匹配成功
        if (i - k == j) and (j == len(t) - 1) and (s[i] == t[j]):
            flag = k
            break
        # s和t相等就继续向后匹配
        if s[i] == t[j]:
            i = i + 1
            j = j + 1

        # 不相等从k的位置开始匹配
        else:
            k = k + 1
            i = k
            j = 0
            # 假如s中所剩字符小于t中所剩字符
            if (len(s) - i) < len(t):
                flag = -1
                break

    return flag


print("请输入描述语句：")
S = input()

for i in range(len(characteristic)):
    if(BF(S,characteristic[i])!=-1):
        describe.append(characteristic[i])

def search(describe):
    for i in range(len(describe)):
        if(describe[i]=="毛发" or describe[i]=="奶"):
            print(describe[i]+"-->"+"哺乳动物")
            if("哺乳动物" not in describe):
                describe.append("哺乳动物")

        if(describe[i]=="羽毛"):
            print(describe[i]+"-->"+"鸟")
            if("鸟" not in describe):
                describe.append("鸟")

        if(describe[i] == "吃肉"):
            print(describe[i]+"-->"+"食肉动物")
            if ("食肉动物" not in describe):
                describe.append("食肉动物")

    if("会飞" in describe and "下蛋" in describe):
        print("会飞，下蛋--->鸟")
        if("鸟" not in describe):
            describe.append("鸟")

    if("犬齿" in describe and "有爪" in describe and "眼盯前方" in describe):
        print("犬齿，有爪，眼盯前方--->食肉动物")
        if("食肉动物" not in describe):
            describe.append("食肉动物")

    if("哺乳动物" in describe and "有蹄" in describe):
        print("哺乳动物，有蹄--->有蹄类动物")
        if("有蹄类动物" not in describe):
            describe.append("有蹄类动物")

    if("哺乳动物" in describe and "反刍动物" in describe):
        print("哺乳动物，反刍动物--->有蹄类动物")
        if("有蹄类动物" not in describe):
            describe.append("有蹄类动物")

    if("哺乳动物" in describe and "黄褐色" in describe and "食肉动物" in describe):
        if("暗斑点" in describe):
            print("哺乳动物，食肉动物，黄褐色，暗斑点--->金钱豹")
            if("金钱豹" not in describe):
                describe.append("金钱豹")
        if ("黑色条纹" in describe):
            print("哺乳动物，食肉动物，黄褐色，黑色条纹--->金钱豹")
            if ("虎" not in describe):
                describe.append("虎")

    if("有蹄类动物" in describe and "长脖子" in describe and "长腿" in describe and "暗斑点" in describe):
        print("有蹄类动物，长脖子，长腿，暗斑点--->长颈鹿")
        if("长颈鹿" not in describe):
            describe.append("长颈鹿")

    if("有蹄类动物" in describe and "黑色条纹" in describe):
        print("有蹄类动物，黑色条纹--->斑马")
        if("斑马" not in describe):
            describe.append("斑马")

    if ("鸟" in describe and "长脖子" in describe and "长腿" in describe and "不会飞" in describe and "黑白二色" in describe):
        print("鸟，长脖子,长腿,不会飞，黑白二色--->鸵鸟")
        if ("鸵鸟" not in describe):
            describe.append("鸵鸟")

    if ("鸟" in describe and "会游泳" in describe  and "不会飞" in describe and "黑白二色" in describe):
        print("鸟，会游泳，不会飞，黑白二色--->企鹅")
        if ("企鹅" not in describe):
            describe.append("企鹅")

    if("鸟" in describe and "善飞" in describe):
        print("鸟，善飞--->信天翁")
        if("信天翁" not in describe):
            describe.append("信天翁")

def result(describe):
    for i in range(len(describe)):
        if(describe[i]=="虎" or describe[i]=="金钱豹" or describe[i]=="斑马"
            or describe[i]=="长颈鹿" or describe[i]=="鸵鸟" or describe[i]=="企鹅" or describe[i]=="信天翁"):
            print("该动物是："+describe[i])

    if("虎" not in describe and "金钱豹" not in describe and "斑马" not in describe
            and "长颈鹿" not in describe and "鸵鸟" not in describe
            and "企鹅" not in describe and "信天翁" not in describe):
        print("无法判断是什么动物！")

print()
print("提取或者总结的特征：")
print(describe)
print()
print("推理过程：")
search(describe)
print()
print("结论：")
result(describe)
