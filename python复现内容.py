# 列表  (1 ~~~~ 100)
lis  = [1, "2", "a", True, False, 1.23]
lis[-1] = "1.23"  # 修改 下标为 -1 的元素
print(lis)        # [1, "2", "a", True, False, "1.23"]

lis[-1] = ["1.23"]  # 修改 下标为 -1 的元素
print(lis)        # [1, "2", "a", True, False, ['1.23']]

lis[-1] = (["1.23"], )  # 修改 下标为 -1 的元素
print(lis)        # [1, "2", "a", True, False, (['1.23'],)]

lis[3:5] = [1.23]  # [True, False] = [float(1.23)]  # 把列表换成可替代元素
print(lis)         # [1, "2", "a", 1.23, "1.23"] 

lis[3:5] = "1.23"  # [True, False] = ["1", ".", "23"]
print(lis)         # [1, "2", "a", "1", ".", "23", "1.23"]

lis[3:5] = ["1.23"]  # [True, False] = [str(1.23)]
print(lis)         # [1, "2", "a", "1.23", "1.23"]  

# 可迭代对象([iterable]) :  string(字符串), list(列表), 元组(Tuple), 集合(Set)    总结 : 除 Number(数字) 外都是可迭代对象.
# 布尔值(bool) :  数字 0, 空值, False, None 都判断为 False
# 返回值 : 对元素进行修改

lis = [1, 2, 3, 4]
lis.append(999)
print(lis)      # [1, 2, 3, 4, 999]

str1 = "12345"
new_s = str1.replace("5", "999")      # 有几个变几个
print(new_s)    # 输出 1234999

lis = [1, 2, 3, 4]
new_l = lis.append(999)
print(new_l)    # None 原因 :  string : 不可变数据类型 -- 无法对原数据进行操作; string --copy-- string --修改-- 新string(返回值) --new_s
print(lis)      #              list   : 是可变数据类型 --- 会对原数据进行改变 ; lis.append([新的内容]) 返回出 -- None(返回值-是个关键字) -- New_l   

lis = [1, 2, 3, 4]
lis.append([[1, 2, 3, 4], [5, 6, 7, 8]])                 # append 可添加列表 在后面添加列表
lis[len(lis):] = [[1, 2, 3, 4]] 
lis.extend([[1, 2, 3, 4]])        
print(lis)

lis.extend([1, 2, 3, 4])        # 添加列表中的元素
print(lis)      #  [1, 2, 3, 4, 1, 2, 3, 4]
lis[len(lis):] = [1, 2, 3, 4]   # 同上
print(lis)      

lis.insert(3, ("物理哇卡"))        # 在第 i 个下标 插入"abc" 内容 , i 可以超过最大索引    可添加字典 , 列表, 元组
print(lis) 

# sort 与 sorted 的差别 :    sort  列表返回值  对原数据操作 , 是列表对象的方法 无返回值; sorted 对所有可迭代对象排序 有返回值 , 是内置函数
# sort([key], reverse = True/False)  排序  没有返回值
lis = [1, -2, 3, -4, False]
lis.sort(reverse = True)
print(lis)         # [-4, -2, False, 1, 3]
# key : 指定一个函数名, 排序之前用完这个函数后进行排序  abs(绝对值)
lis.sort(key = abs, reverse = False)   # 把lis 中的每个元素, 作为abs 的参数, 然后调用该函数, 相当于 abs(这里面为lis 的每个元素)
print(lis)

sorted(lis, key = abs, reverse = True)    # lis 改变   以列表形式返回  sorted (列表 , key = 调用函数, reverse = F/T)   
lis.sort(reverse = True)                  # lis 不变   以列表形式返回

#  list.reverse  和  reversed 区别  
#  序列 :  字符串 , 元组 , 列表  
lis = [1, 5, 7, 2, 1]
lis.reverse()        # 只针对列表
print(lis)        # lis顺序反过来

list(reversed(lis))    # 针对序列       # 迭代器 <list_reverseiterator object at 0x000001B3DBAE0100>

str1 = ([5, 9, 9, 8, 7], [5, 8, 7, 6, 7], [2, 3, 9, 8, 7])  # 排序的时候按照第一个从大到小排序 第一个相同则第二个排序 以此类推
res = (reversed(str1))
res2 = list(res)   
print (res2)

str1[0].count(9)  # 查找某个元素出现的次数

list.index(x, start, end)     # x : 要找的值 ; start : 开始的位置; end : 结束的位置

a = [1, 2, 3, 4, 3, 2, 3]
print(a.index(2, 2, 8))    # 输出为第二个2 检索第一个2的位置 只能检索到第一个

# list.pop(X)  X 为索引下标
li = [1, 2.3, 2+3j, "4", True, False]   # 会返回删除的元素
li.pop()      # 拿走了 false
li.pop(2)     # 拿走了搜索引位置的数据 
item = li.pop(2)  # 取出了索引为 2 的内容
print(item)

# remove 没有返回值
iteam = li.remove(2.3)   # [1, 2+3j, "4", True, False] 
print(iteam)  # None

lis = [1, 2, 3, 4]
lis2 = lis.copy()    # 浅复制一次表lis
lis3 = lis[:]        # 等效于copy

lis2.clear()          # 清空列表
del lis2[:]           # 同上

# 元组  (102 ~~~  111)
tup1 = (1, 2, 3, 4)       # 元组
lis1 = [1, 2, 3, 4]       # 列表

tup = ((((123), )), )
print(tup)

# 封包
tup = 1, 2, 3, 4, 5
print(tup)

# 字典  (113 ~~~ 243)
dic = {'name': "张三", 'age': 18, 'eat': "鱼"}     # dic{key: "内容"}   key : 键 ---不可变的 所以列表就不行  键重复 后面的内容会覆盖前面的内容; 内容随意
dic = {(1, 1, 3, 4): (1, 2, 3, 4)}
print(list(dic))            # 只对key 列表输出

# 创建字典的6个方式
# 1、 直接在空字典添加
dic = {'name': "张三", 'age': 18, 'eat': "鱼"}     
dic = {(1, 1, 3, 4): (1, 2, 3, 4)}
# 2、定义一个空字典 ,之后添加键值对
dic = {}
dic['name'] = "张三"
dic['age'] = 18
dic['eat'] = "鱼"
print(dic)
# 3、把键作为关键字传入  dict(**kwarg)   kwarg : 关键字参数    j键值对 : A=B
print(dict(name="张三", age=18, eat="鱼"))
# 4、迭代对象的方式   dict(*iterable)         * : 表示不定长参数  可以输入多个对象   返回一个元组迭代器
print(dict([['name', '张三'], ['age', 18], ['eat', '鱼']]))
# 5、通过zip 打包  dict(mapping)    zip(*iterable)  
print(dict(zip(["name", "age", "eat"], ["张三", 18, "鱼"])))    # 各组里少一个都会消失  dict 只能是二维  ; list 可以多维
print(list(zip(["a", "b", "c"])))            
# 6、fromkeys(可迭代对象, 值)  类class 方法    有局限性   作用: 创建一个新的字典
dic = dict.fromkeys([1, 2, 3, 4], 'abcd')
print(dic)      


# 字典的对象方法
# dict.keys  通过字典的键组成一个新视图
dic = {"name": "Tom", "age": 18, "weight": 75}
res = dic.keys()
print(dic)      #   {'name': 'Tom', 'age': 18, 'weight': 75}
print(res)      #   dict_keys(['name', 'age', 'weight'])

lis = list(res)   # res 前处理lis
print(lis)       #   ['name', 'age', 'weight']

dic["hight"] = 185
print(dic)      #  {'name': 'Tom', 'age': 18, 'weight': 75, 'hight': 185}
print(res)      #   dict_keys(['name', 'age', 'weight', 'hight'])
print(lis)      #  ['name', 'age', 'weight']

# 对比原因 : 
dic = {"name": "Tom", "age": 18, "weight": 75}  # 原字典
res = dic.keys()                #  原字典对象              res 随 dic 的改变而改变
print(res)                      #  修改前的 res 
print(list(res))                #  ['name', 'age', 'weight'] 
lis = list(res)                 #  由改变前的res 得到lis ---|    (157 → 158)       生成新的lis 空间    
print(lis)                      #  输出lis ----------------|

dic["hight"] = 185              # 修改后 res 将发生改变 多 "hight" key              
print(res)                      # 输出修改后的 res
print(list(res))                # ['name', 'age', 'weight', 'hight']
print(lis)                      # 'name', 'age', 'weight']     最后这个lis输出与158行 一样的原因 :    lis 为新生成新空间 而 res改变后 lis 不随新的res的改变而改变

# dict.values  通过字典的值组成一个新视图
d1 = {"身高": 175, "体重": 65, "肤色": "黑色", "名字": "张三"}
a = d1.values()
print(a)        # dict_values([175, 65, '黑色', '张三'])
print(list(a))  # [175, 65, '黑色', '张三']
b = list(a)
print(b)        # [175, 65, '黑色', '张三']

d1["肤色"] = "黄色"
print(a)        # dict_values([175, 65, '黄色', '张三'])
print(list(a))  # [175, 65, '黄色', '张三']
print(b)        # [175, 65, '黑色', '张三']

# dict.items()  # 通过一个键值对(key, value)来组成新的视图
d1 = {"身高": 175, "体重": 65, "肤色": "黑色", "名字": "张三"}
a = d1.items()
print(a)        # ict_items([('身高', 175), ('体重', 65), ('肤色', '黑色'), ('名字', '张三')])
print(list(a))  # [('身高', 175), ('体重', 65), ('肤色', '黑色'), ('名字', '张三')]
b = list(a)
print(b)        # [('身高', 175), ('体重', 65), ('肤色', '黑色'), ('名字', '张三')]

d1["体重"] = 77
d1['性别'] = "女"
print(a)        # dict_items([('身高', 175), ('体重', 77), ('肤色', '黑色'), ('名字', '张三'), ('性别', '女')])
print(list(a))  # [('身高', 175), ('体重', 77), ('肤色', '黑色'), ('名字', '张三'), ('性别', '女')]
print(b)        # [('身高', 175), ('体重', 65), ('肤色', '黑色'), ('名字', '张三')]

# dict.get(key, default) key : 键值  ; default : 返回值, 指定键不存在 返回该值 , 默认为None
d1 = {"身高": 175, "体重": 65, "肤色": "黑色", "名字": "张三"}
d1.get("体重")   # 获取key中的信息
value = d1.get("体重")
print(value)

value = d1["性别"]
print(value)    # key 里没有性别 会报错
value = d1.get("性别", "None")
print(value)    # None  

# dict.update([other])     用other 的键 or 值 更新字典    other 是可迭代对象
# 1、更新键 
d1 = {"身高": 178,"名字": "张三"}
d2 = {"肤色": "蓝色"}
d3 = {"身高": "1米78"}

d1.update(d2)
d1.update(d3)
print(d1)       # {'身高': '1米78', '名字': '张三', '肤色': '蓝色'}

d1.update([("name", "王舞")])       # 加入一个键值对
d1.update(身高=178)   # 键相同后面的值会覆盖前面的值
d1.update(zip(["名字"], ["灵剑"]))
print(d1)

# dict.pop(key, default)  移除指定的键   key : 键值  ; default : 指定键不存在 返回该值 , 默认为None
d1 = {"身高": 178,"名字": "张三"}
value = d1.pop("名字")   # 在原数据中删除
print(value)
print(d1)
res = d1.pop("名字", "没有了")
print(res)

# 移除最后一个键值对 返回构成他们的元组
p_value = d1.popitem()
print(p_value)  # ('名字', '张三')
print(d1)       # {'身高': 178}

# dict.setdefault(key. default)    如果有key 则获取值 ;  如果不存在, 则返回default
d1 = {"身高": 178,"名字": "张三"}  
value = d1.setdefault("身高", 175)
print(value)

value = d1.setdefault("weight", 175)    # 如果不存在, 会在原数据添加键值对
print(value)

d1.setdefault("身高", 175)   # 原数据不发生改变
d1

# 题注 (245 ~~~ 248)
# dic.copy 复制  字典是可变的 集合是可变的 列表是可变的  推出👉可变的能复制  增删改 对原数据进行操作  
# dic.clear 清空 ,清空列表  == del a[:] 清空
# 序列 : 字符串, 列表, 元组  通过索引和切片的方式访问元素

# del 语句  不影响引用的值  三要素: 变量名 值 变量名指向值 (250 ~~~~ 269)
a = 999 
b = a   # 999 引用为2
del a   # 只删除了a, a不再指向999;  b 创造了新的空间  b还在

a = [-2, 0, 1, -3, 66.66, 333, 666, 777,88.8]
del a[0]
print(a)    # 删除了 -2

del a[2:4]
print(a)

del a[:]
print(a)  #  == a.clear

d1 = {"身高": 175, "体重": 65, "肤色": "黑色", "名字": "张三"}
del dic["age"]   # 根据key 删除
print(d1)     

# 字典不存在清空 只能 dic.clear

# 集合  (271 ~~~ 291)
# set 集合 性质 : 1.可改变 , 不是序列 ; 2.无序性 : 内容随机排序; 3.不重复性; 4.只能包含不可变数据类型   字典是可变的 集合是可变的 列表是可变的
# 创建空集合  
set((1, 2, 3, 4, 5, 6))
set([iterable])  # iiterable 可迭代对象 , 除数字外 其他都是

# 冻结集合    frozenset([iterable])   是不可变集合
print(frozenset([1, 2, 3, 4, 5, 6]))        # 不会对元素进行修改  , 无序的  适合集合所有性质
print(frozenset({1: 2, 3: 4}))     # 只输出键 {1 ,3}

# 关系测试
a = set("abdefgaagez")
b = set("abcd")
c = set("abfz")
print(c <= a)   # True  判断 c 是不是 a 的子集 返回 bool 值
print(a & b)    # {'a', 'd', 'b'}  判断 a 和 b 的交集
print(a | b)    # {'c', 'e', 'b', 'f', 'd', 'z', 'a', 'g'}  判断 a 和 c 的并集
print(a - c)    # {'d', 'e', 'g'}   计算 a 和 c 的差集
print(a ^ b)    # {'e', 'g', 'f', 'z', 'c'}    对称差
print(a ^ b)                            #-------------------------------------------------|
print((a | b) - (a & b))                #(a ^ b) == ((a | b) - (a & b))-------------------|

# 集合(set)的对象方法       (293 ~~~ 356)
# set.isdisjoint(iterable)   判断是不是没有交集
set1 = {1, 2, 3, 4, 5}
frozenset1 = frozenset({1, 2, 3, 4, 5, 6})

set1.isdisjoint([5, 6, 7])     # False  判断是否有交集, 有交集为False 无交集喂True 
print(frozenset1.isdisjoint("1234"))   #  True

# issubset(iterable)        判断是不是子集     对应运算符 "<="
print(set1.issubset(frozenset1))          # 判断外面的是不是里面的子集  外包内 是 True  否 False

# issuperset 
print(set1.issuperset(frozenset1))        # 判断里面的是不是外面的子集  内包外 是 True  否 False  里面含有元组 元组里的数字与上面相同 也算不包含

# union(*iterable)     * 可传多个 可迭代对象   联合 取并集 返回新的集合
set1.union(frozenset1, ["a", "b"])        # {1, 2, 3, 4, 5, 6, 'a', 'b'}

# intersection(*iterable)       取交集      可多个iterable
set1.intersection(frozenset1, [1, 4, "a", "b"])       # 求出共有的元素

# difference(*iterable)       取差集  找出不同元素  可多个iterable
set1 = {1, 2, 3, 5}
frozenset1 = frozenset({3, 4})
set1.difference(frozenset1, [1, 4, "a", "b"])         # {2, 5}    找出两个里面有在set1 不在 frozenset1 的数    

# symmetric_difference(iterable)    对称差    并集减去交集  只要求一个参数  把独有的取出来
print(set1.symmetric_difference([1, 3, "a", "B"]))    # {2, 4, 5, 'a', 'B'}  得出不是共有的元素

# copy  返回集合的浅拷贝

# 集合特有
# set.update(*iterable)   更新内容  重复的会去除(集合的不重复性)
set1 = {"1", "2", 1, 3}
set1.update([1, 4, 2], "abcd")
print(set1)

# intersection_update(*iterable)    会对元素进行操作, 取交集
set1.intersection_update([1, 4, 2, "1"], "12abcd")
print(set1)             # 共有的元素为 {1} 

set1.difference_update([1, 4, 2, "1"], "12abcd")      # 会对元素进行操作, 取差集   取差集  由 set 里的减去给定条件里的
print(set1)     # {3}

set1.symmetric_difference_update("12abcd")      # {1, 2} 共有部分  取对称差 并集减去交集
print(set1)     # {'1', 'b', 3, 'd', '2', 'a', 'c'}   # 剔除共有部分后

# set.add (other)   把其他的加入 set    集合中不能放可变的数据类型
set1.add(11, 2, 3)                # 只能放一个  不能放可变数据类型  --------------------------------| 对
set1.update("",)                  # 可放多个    放里面的元素   放列表会加入元素 ---------------------| 比

# set.remove(elem)         移除集合中的一个元素  没有报错
set1.remove(1)

# set.discard(elem)       移除元素, 如果不存在不报错
set1.discard(1)
print(set1)

# set.pop()    从集合中移除一个 并返回
set1 = {1, 2, 3, 4, 5}
set1.pop()
print(set1)             # 无序的 随机删除

# set.clear 清空集合
set1.clear()
 
# 赋值      (358 ~~~ 383)
# 变量的类型 : 变量地址上存放的值的类型
a = 999
print(a)

# 查询变量地址 id(变量)
a = 999
id(a)             #  a新空间的存储的 999 ------| 不是一个999
id(999)           #  原来999 储存的空间--------|
"""终端为什么会相同? """    # 答 : 大整数式对象 解释器做了优化 节省空间  所以终端输出地址相同 , 小整数式[-5, 256]不会销毁, 一直可以用

a = 999     #  ---------| a 和 b是两个999 ----------| 
b = 999     #  ---------|                          |
c = a       #  ------------------------------------| a 和 c 的999相同

del a       # 解除之后 c的id 不变
print(id(c))

"""如果变为列表 [999, 888, 777] 整个运行过程结果不变"""
a = [1, 2, 3]
b = [1, 2, 3]
c = a
a.insert(1000, 4)
print(a)     #  a, c 发生改变 原因 : 数据地址相同  原数据发生改变, 相同地址的会跟着变
print(b)
print(c)

# 深拷贝  浅拷贝     (385 ~~~ 446)
# 作用 : 建立副本, 对副本进行操作
# 对于拷贝内容, 只对可变类型数据拷贝    前面所学都是浅拷贝
import copy
import traceback

a = [1, 2, 3]
b = a.copy()
b

"""
浅拷贝 : copy.copy()
深拷贝 : copy.deepcopy()
"""
# 浅拷贝 : copy.copy()
[1].copy()      #  ------[1] 代表列表   还可以用 set() , 字典{ }
[1][:]
copy.copy((1, 2))  #   这种可以拷贝元组
# 深拷贝 : copy.deepcopy()
copy.copy({})

# 浅    只拷贝可变数据类型的外皮, 不考虑内部的单个元素  只考虑本身
tup1 =  (991, "abc")
tup2 = copy.copy(tup1)
print(id(tup1))
print(id(tup2))             # 浅拷贝没有复制出来

tup1 =  (991, "abc", [123])   # 哪怕里面有列表 元组也是不可变数据类型
tup2 = copy.copy(tup1)
print(id(tup1))
print(id(tup2))             # 浅拷贝只拷贝出来一个皮儿~~

# 深    拷贝全部的可变数据类型, 考虑内部的各个储存元素的地址, 也包含内部各个元素的各个元素。
tup1 = (991, "abc",[])
tup2 = copy.deepcopy(tup1)
print(id(tup1))
print(id(tup2))     # 地址和上面不同    原因 : 里面有不可变数据类型 []

tup1 = (991, "abc",((({}, ), ),))
print(tup1[-1][0][0][0].tupdate({1: 2}))

lis1 = [991, "abc", (9, 993), [994, 995], [888, 887], {"name": "Tom"}, (996, [997, 998]), (888, (886, 886))]
lis2 = copy.copy(lis1)   # 浅拷贝后里面数据全部指向同一个地址
lis3 = copy.deepcopy(lis1)    # 深拷贝后对里面可变类型数据的各个元素也进行了拷贝   复合元素有可变  会把该对象新拷贝一份

print(id(lis1))
print(id(lis3))    # 地址不同

lis1.append(9)
print(lis1)
print(lis3)        # lis1 和 lis3 元素不同     原因 : 指向不同内存, 地址不同

lis1[3].append(999)     # 只加列表1 ,   3 不加
print(lis1)
print(lis3)

lis1[5].update(age = 18)    # 只变1  不变3  地址不一样
print(lis1)
print(lis3)

lis1[6][-1].pop()   # 只变1  不变3  地址不一样
print(lis1)         
print(lis3)

# 运算符、优先级 (448 ~~~ 471)
a = 1.5
b = 22
c = 333
print(a + False)    # 加
print(a - b)        # 减
print(a * b)        # 乘
print(a / b)        # 除
print(a ** b)       # 指数为b 底数为a
print(b // a)       # 取整      向下取整                数轴为例 : 向后取整
print(int(b / a))   # 取整      向上取整
print(8 % -3)       # 取模      取余数              8 - 3*3 = -1
print(-8 % 3)       #                              -8 + 3*3 = 1
print(-21 % 8)      #                              -21 - 8*2 = 5

# 运算符优先级
a = 90
b = 90.5
res = a > b         # 判断a > b
res = a == b        # 判断 a 恒等于 b
print(a >= b)       # 判断 a >= b
print(a <= b)       # 判断 a <= b
print(a < b)        # 判断 a < b
print(a != b)       # 判断 a 不等于 b

# 赋值运算符        (473 ~~~ 481)
# 左边为原值和结果值(原值经过计算👉得到新结果👉返回原值)   右边为所要计算的数值
a = 64
a += 3              # 基本理解为 a + 3 = a    
a -= 3              # 基本理解为 a - 3 = a
a *= 4              # 基本理解为 a * 4 = a         
a /= 1.5            # 基本理解为 a / 1.5 = a
a //= 4             # 基本理解为 a // 4 = a   整除
a **= 8             # 基本理解为 a ** 8 = a

# 拼接              (483 ~~~ 515)
lis1 = [1, 2, 3, 4]
lis2 = [4, 5]
lis1 = lis1 + lis2 
print(lis1)         #  改变了s1的值

lis1 = [1, 2, 3, 4]
lis2 = [4, 5]
lis100 = lis1
lis1 = lis1 + lis2  #  这个有返回值 只改变了lis1 的结果
print(lis1)         #  [1, 2, 3, 4, 4, 5]    返回了新的结果
print(lis100)       #  [1, 2, 3, 4]    没有返回新的结果

lis1 = [1, 2, 3, 4]
lis2 = [4, 5]
lis100 = lis1
lis1 += lis2        #  这个没有返回值 直接改变原数据 所以所有引用的值都会发生改变
print(lis1)         #  [1, 2, 3, 4, 4, 5]    返回了新的结果
print(lis100)       #  [1, 2, 3, 4, 4, 5]    返回了新的lis1

tup1 = (1, 2, 3)
tup2 = (4, 5, 6)
tup100 = tup1
tup1 = tup1 + tup2
print(tup1)         #   (1, 2, 3, 4, 5, 6)
print(tup100)       #   (1, 2, 3)

tup1 = (1, 2, 3)
tup2 = (4, 5, 6)
tup100 = tup1
tup1 += tup2
print(tup1)         #   (1, 2, 3, 4, 5, 6)
print(tup100)       #   (1, 2, 3)   未变的原因 : 元组为不可变类型数据

# 增强赋值语句      (517 ~~~ 525)
"""
普通赋值      以新建方式进行赋值  有一个新的返回值 copy一个新的值
增强赋值      直接对原数据进行修改 以追加的方式进行处理 效率比普通的高
"""
lis1 = [1, 2, 3]
lis2 = [4, 5]
lis1 = lis1 + lis2   # 并不是把lis1 的拿出来 而是复制一个lis1 然后把 lis2 的元素放进去 返回新的值
lis1 += lis2         # 直接修改lis1 直接加上lis2 

# 序列赋值          (527 ~~~ 537)
a, b, c = 1, 2, 3    # 每个元素一一对应
print(a)
print(b)
print(c)

[a, b, c, d] = (123, [1, 2, 3, 4, 5, 6], {1, 2, "a", b}, (2, 3, 4, 5, 6, [1, 2, 3, 4, 5]))
print(a)
print(b)
print(c)             # 同样一一对应
print(d)

# 封包              (539 ~~~ 541)
a = 1, 2, [3, 3, 3], 4
print(a)

# 多个赋值          (543 ~~~ 547)
a = b = c =999
print(id(a))
print(id(b))
print(id(c))         #  地址相同

# 海象运算符        (549 ~~~  564)
"""写法一"""
string = "王舞爱修仙"
length = len(string)
print(length + 5)
print(f"string的长度为:{length}")

"""写法二       简化了一"""
string = "王舞爱修仙"
print(len(string) + 5)
print(f"string的长度为:{len(string) + 5}")

"""写法三       对二的输出长度更加明确"""
string = "王舞爱修仙"
print((length := len(string)) + 5)
print(f"string的长度为:{length}")

# 逻辑运算符        *(566 ~~~ 597)
"""bool 判断(not and or)"""   #  False 为假  True 为真  
a = 61
b = "这里是world"
c = []
d = 0           
# and 有一个假 则假; 有两个真 为真     左边为真返回右边 
print(a and b)              # 这里是world       都为真返回and 右边的值 
print(b and a)              # 61          
print(c and d)              # []                都为假返回and 左边的值
print(d and c)              # 0
print(a and d)              # []               一真一假则假

# or 有一个真 则真; 有两个假 则假      左边为真返回左边
print(a or b)               # 61            都为真 返回左边
print(b or a)               # 这里是world             
print(c or d)               # 0             都为假 返回右边
print(d or c)               # []
print(b or c)               # 这里是world    一真一假返回真

# not 否定他的对错
print(not a)                # False   判断不是a 的情况
print(not c)                # True    判断不是c 的情况

# 优先级 (not> and > or)
print(not a or b and c)          # 先not a----False👉----d and c----[](原因 : 取右边的)👉----False or []----👉[]

# 短路原则     (593 ~~~ 597)       前面已经计算出表达式结果后面不会再计算
print(b or a / d)                # 这里是world    原因 :  1、res = a / d ; 2、 b or res ; 3、输出 b。 因为b 与res没有关系 所以为了提高效率直接输出 b
print(c or a / d)                # 报错    原因 :  1、res = a / d ; 2、 c or res 中c为false 需要res判定, 因为res 无法存在; 3、报错
print(b or a / b and a / d)      # 理论上先执行 and 但是 b 为真 所以 为了提高效率直接输出 b  原因 : b 和res 无关
print(1 - (1 != 9) and "1" + 1)  # 左边为假短路原则, 后面报错不显示

# 成员运算符   (599 ~~~ 604)
# 判断 某个元素是不是在对象当中    in  在里面  ;  not in 不在里面
lis1 = ["56", 56, [1, 2], 61, True]
print(1 in lis1)                 # 1 为 True  在里面
print(2 in lis1)                 # 列表中的列表元素算一个大项, 表里的表里的元素有那一项不能算成员, 他也没在里面
print("a" not in lis1)           # 判断元素a 不在列表中

# 身份运算符   (606 ~~~ 620)
# 判断两个标识符是不是引用自同一个对象; is  判断地址是否相同 id(a) = id(b)  ; is not 判断地址是否不同 id(a) != id(b)
a = 999 
b = 999
# is 比较地址是否相同, 如果地址相同, 数值一定相同; is not 相反  判断地址是不是不同
print(a is b)                    # False  原因: id(a) 不等于 id(b)   
print(a is not b)
# 比较数值大小是否相同
print(a == b)
# 比较地址是否相同   is not 相反
print(id(a) == id(b))

a = b = c = 999
print(a is b is c)              # 可多者比较

# 位运算符      (621 ~~~ 703)
# 位运算符 : 把数字看做二进制来进行计算
""" x 为任一实整数"""
bin(x)      # 转换为二进制
hex(x)      # 转为十六进制
oct(x)      # 转换为八进制

bin(61)      # '0b111101'  输出结果 0b + 二进制数字
hex(61)     # '0x3d' 输出结果 0x + 十六进制数字
oct(61)     # '0o75' 输出结果 0o + 八进制数字

print(bin(-61))    # '-0b111101' 负数输出结果为 -0b + 二进制数字
print(hex(-61))    # '-0x3d' 输出结果 -0x + 十六进制数字
print(oct(-61))    # '-0o75' 输出结果 -0o + 八进制数字                                                 

# 有符号整数(第一位代表正负号) and 无符号整数(从零开始, 没有负数 第一位不代表正负号)   
"""
1、有符号整数(正负之分) 按位进行操作  : 符号位为 0 代表正数 , 符号位为 1 代表负数 一般在第一位; 符号位不参与计算
2、无符号整数(从零开始, 没有负数 第一位不代表正负号) 有多少个算多少个
3、二进制运算 ： 一串二进制数字  为0 就是 0 的 所在位置次方  1 为 2 的所在位置次方  从最后一位【0】开始
"""

# 按位与运算
print(25 & 61)    # 25    结果和下面一样---------|                   -----|  25	 👉二进制   0	0	0	1	1	0	0	1    |
print(61 & 25)    # 25    结果和上面一样---------|                   -----|  &                                               |
bin(25)           # 0b11001                           		        -----|  61	👉二进制   0   0   1   1   1   1   0   1    |
bin(61)           # 0b111101                                        -----|  得到 :                                          |
# 按位与运算符& : 1 遇 1 为 1 ; 1 遇 0 为 0 ; 0 遇 0 为 0             -----|  25  👉二进制   0   0	 0   1   1   0   0   1    |

# 按位或运算												
print(25 | 61)    # 25    结果和下面一样---------|                   -----|  25	 👉二进制   0	0	0	1	1	0	0	1    |
print(61 | 25)    # 25    结果和上面一样---------|                   -----|  |                                               |
bin(25)           # 0b11001                           		        -----|  61	👉二进制   0   0   1   1   1   1   0   1    |
bin(61)           # 0b111101                                        -----|  得到 :                                          |
# 按位或运算 | : 1 遇 1 为 1 ; 1 遇 0 为 1 ; 0 遇 0 为 0              -----|  61  👉二进制   0   0	 1   1   1   1   0   1    |													
													
# 按位异或运算	 不同为1 相同为0											
print(25 ^ 61)    # 25    结果和下面一样---------|                   -----|  25	 👉二进制   0	0	0	1	1	0	0	1    |
print(61 ^ 25)    # 25    结果和上面一样---------|                   -----|  |                                               |
bin(25)           # 0b11001                           		        -----|  61	👉二进制   0   0   1   1   1   1   0   1    |
bin(61)           # 0b111101                                        -----|  得到 :                                          |
# 按位异或运算 ^ : 1 遇 1 为 0 ; 1 遇 0 为 1 ; 0 遇 0 为 0            -----|  36  👉二进制   0   0	 1   0   0   1   0   0    |

"""
负数的源码 : 是正数的补码
负数的反码 : 除符号位外, 其他反过来 0 变 1, 1 变 0;    下面顺序 :正  反  补  源
exp :  (0  0  0  1  1  0  0  1), (1  1  1  0  0  1  1  0), (1  1  1  0  0  1  1  1 ), (1  0  0  1  1  0  0  1),
负数的补码 : 除符号位外, 其他取反码, 在最后一位 +1 
正数的反码和补码 都与源码相同
补码 - 1 = 反码   ;  反码 + 1 = 补码
"""

# 右移(>>)   左边补对应的符号位置! 补0
print(25 >> 2)          # 6                                          -----|  25  👉二进制    0  0  0  1  1  0  0  1   |
""" 25 右移两格"""       # 得到 ( 1 1 0)(2进) 👉 6 (10进)             -----|  6   👉二进制    0  0  0  0  0  1  1  0   | 0  1

# 负数推进按位运算过程 符号位补1: 源码 👉 反码 👉 补码 👉 推进两格 👉得到结果----👉 补码(不够借位) 👉 反码 👉 负数原码
print(-25 >> 2)         # -7  负数以源码的补码表达                     -----| -25  👉二进制    0  0  0  1  1  0  0  1   |            
""""-25 右移两格 """     # 得到 ( 1 1 1)(2进) 👉 -7 (10进)            -----| -7   👉二进制    0  0  1  0  0  1  1  1   | 0  1

# 左移(<<)   右边补对应的符号位置! 补0
print(25 << 2)          # 100                                        -----|  25   👉二进制         0  0  0  1  1  0  0  1   |
""" 25 左移两格 """      # 得到 100 翻四倍                             -----|  100  👉二进制   0  0  0  1  1  0  0  1  0  0   | 

# 负数推进按位运算 : 过程同右移 符号位补1
print(-25 << 2)         # 得到 -100   翻 -4 倍   

# 心得 : 二进制右侧添 2 个 0 表示 原数扩大四倍

# 取反(~)  符号位都反 1 变 0 , 0 变 1  ;对 x 取反  为  -(x + 1) ;
print( ~25 )            # -26   正数取反      -----|  25 取反 →   1  1  1  0  0  1  1  0  (反后码)  → 需要推算回去 → (补码 - 1 = 反码) 求出反码 → 除了符号位其他都反过来   |
print( ~(-25) )         #  24   负数取反      -----| -25 取反 →   0  0  0  1  1  0  0  1  (原码)  → 求反码 → 求补码 → 对补码反码 → 不用再推回去了                         |

print(25 >> 7)          # 正数  推 7 位 只剩 0
print(-25 >> 7)         # 负数 推 7 位 只剩 -1

"""
如果按位操作的数据是正数可以直接操作, 因为他的反码补码都一样 ; 
如果按位操作的数据是负数那么要先求出他的补码, 再按位操作。
如果按位操作之后的数据是正数, 直接转成十进制就是结果
如果按位操作之后的数据是负数, 需要先求出他的原码, 再转成十进制
"""

# +、 +=、 * 的连接操作  (704 ~~~ 729)
# 要保证连接对象一直 字符连字符 列表连列表  元组连元组
tup1 = (1, 2, 3, 4)
tup2 = (5, 6, 7, 8)
tup1 + tup2
tup1 += tup2   # 元组不可变 等于 t1 + t2
print(tup1) 

lis1 = [1, 2, 3, 4]
lis2 = [5, 6, 7, 8]
lis1 + lis2 
lis1 += lis2   # 对lis1 原储存数据改变 
print(lis1)

tup1 * 3    # 把里面元素复制了三遍 列表同理 字符串说也一样

# 进制关系
"""二进制三格子等于八进制一格,[0 0 1]1 [0 1 0]2 [1 0 1]5  得到125"""
"""二进制四格子等于十六进制一格,[0 1 0 1 ]5 [0 1 0 1]5    得到55""" 
"""八进制转十六进制  →  先转到 10 进制 再到 16  123 得到53,  八进制 : 123  十进制 : 1*(8**2) + 2*8**1 + 3*(8**0) = 83"""
"""十进制转十六进制  →  83/16 = 5 ----3      83/5 = 0----3   得到53""" 
 
# 序列赋值
list1 = [ 1, 2, 3, 4]
list1[0], list1[-1] = list1[-1], list1[0]  # 把 第一个和最后一个对调
print(list1)

# 条件语句     (730 ~~~ 738)
ans = input("老师, 请问线下复工了吗?(Y/N/UN): ")
if ans == "Y":
    print("我明天去线下上课")
elif ans == "UN":
    print("问问其他同学")
else:
    print("好的, 我再等通知.")

# 三元表达式    (740 ~~~ 796)
score = float(input("你这次考了多少分? :"))
if score >= 90: 
    print("你很优秀")
elif score >= 80:
    print("你很不错")
elif score >= 60:
    print("你还行")
else:
    print("时代还在进步, 你还仍需努力")

if score >= 90: 
    result = "你很优秀"
elif score >= 80:
    result = "你很不错"
elif score >= 60:
    result = "你还行"   
else:
    result = "时代还在进步, 你还仍需努力"
print(f"我今天考了{score}分, 老师说{result}")

# 转三元表达
print("你很优秀") if score >= 90 else print("你很不错") if score >= 80 else print("你还行") if score >= 60 else print("时代还在进步, 你还仍需努力")
"""结果 + 条件 + 判断前面为否结果 + 判断条件 + 判断前面为否的结果 + 判断条件 ------------"""
print("你很优秀") if score >= 90 else \
print("你很不错") if score >= 80 else \
print("你还行") if score >= 60 else \
print("时代还在进步, 你还仍需努力")

result = "你很优秀" if score >= 90 else \
"你很不错" if score >= 80 else \
"你还行" if score >= 60 else \
"时代还在进步, 你还仍需努力"
print(f"我今天考了{score}分, 老师说{result}")

# 海象运算符用在成绩位置
print(f'今天我考了{(score := float(input("输入你这次考试的成绩: ")))}分, 老师对我说: {"你很优秀" if score >= 90 else "你很不错" if score >= 80 else "你还行" if score >= 60 else "时代还在进步, 你还仍需努力"}')
# 海象运算符用在判断条件的第一个
print(f'今天我考了{score}分, , 老师对我说: {"你很优秀" if (score := float(input("输入你这次考试的成绩: "))) >= 90 else "你很不错" if score >= 80 else "你还行" if score >= 60 else "时代还在进步, 你还仍需努力"}')

# 嵌套语句  (780 ~~~ 797)
ans1 = input("你是送快递的么: ")
if ans1 == "是":
    print("放在门口")
else:
    ans2 = input("那你是做什么的, 是送外卖的吗? ")
    if ans2 == "是":
        print("挂着门上就行")
    else:
        ans3 = input("你到底是做什么的? ")
        if ans3 == "送蛋糕的":
            print("开门, 拿蛋糕。")
        else:
            print("请听到滴声后留言")

# 三元表达式
print("放在门口") if input("你是送快递的么: ") == "是" else print("挂着门上就行") if input("那你是做什么的, 是送外卖的吗? ") == "是" else print("开门, 拿蛋糕。") if input("你到底是做什么的? ") == "送蛋糕的" else print("请听到滴声后留言")

# 注: 一个 if 一个结果 exp:         (798 ~~~ 812)
score = int(input("输入你的成绩: "))
if score >=90:
    print("掌握的还行")
elif score >= 80:
    print("还不错")
if score >=60:
    print("学的时候继续巩固")
else:
    print("回炉再来")

if 123456:
    print("12345判断为True" )
else:
    print("判断为False")

# 判断bool    (814 ~~~ 827)
if {} :           
    print("{}为False")     # 他会进行bool判断 根据 内容进行判断是否是True 或者 False ; 如果判断是True 则 print 第一个结果; 如果为False 则 pring 为else 的结果 或者继续运行
else:
    print("{}为True")


# 进行bool判断时, None/0/空[字符串, 列表, 元组, 字典, 集合]/False 为False, 其他为True
if {} and {1}:             # 判断为False 输出1234
    print(1234)
elif {1} or {}: 
    print(1)               # 否则为1
else:
    print()

# 导入 random 模块    (829 ~~~ 848)
import random
random.randint(1,4) # 返回 范围内的随机整数
random.random( )    # 返回[0.0, 1.0] 范围内随机浮点数 
random.uniform()    # 返回[a, b] / [b, a] 范围内的随机浮点数
random.choice(seq)  # 从非空序列返回一个元素, 序列 : 列表, 元组, 字符串
random.sample((1, 3, 5, 6, 7), i)    # 随机抽 i 个元素 

lis1 = [1, 2, 3, 4, 5] 
random.shuffle(lis1)    # 打乱内容
print(lis1) 
 
# 范围    range(后面有变量就是开始.没有其他就是结束, 结束位置, 步长)
range(9)   # 得到一个序列 
list(range(9))    # [0, 1, 2, 3, 4, 5, 6, 7, 8]
list(range(7,21,7))   # 7 开始 21 结束 步长为7 结果为[7, 14] ;  21 取不到, 原因 : 末尾取数为 n-1

random.randrange(1, 100, 3)     # 从range里面取一个元素   从 1 - 100 里面 取一个步长为3的数

print(random.randint(2, 5))

# 固定    (850 ~~~ 863)
random.seed(0)                  # 里面数字为数字代号, 没有什么实际意义, seed会固定它下面所有random 随机数
print(random.randint(2, 5))
print(random.randint(2, 5))

# 交叉固定
random.seed(0)   
print(random.randint(2, 5))     # 固定一
random.seed(2)  
print(random.randint(2, 5))     # 固定二
random.seed(2)  
print(random.randint(2, 5))     # 由固定二固定
random.seed(0)  
print(random.randint(2, 5))     # 由固定一固定

# 猜拳游戏      (864 ~~~ 879)
computer = random.choice(("石头", "剪刀", "布"))
print(computer)

INFO = ("石头", "剪刀", "布")
player = int(input("请出拳(0 代表石头/1 代表剪刀/2 代表布): "))
computer = random.randint(0, 2)
print(f'玩家出拳: {INFO[player]}')
print(f'电脑出拳: {INFO[computer]}')

if player == computer:
    print("平局")
elif player - computer == -1 or player - computer == 2:
    print("玩家获胜")
else:
    print("电脑获胜")

# 循环语句      (881 ~~~ 924)

a = 10
while a > 3:
    a -= 3
    print(a)
    if a < 0:
        break

a = 0
while True:
    print(a)
    a += 2
    if a > 100:
        break

count = 0
while count < 5:
    print(count, "小于5")
    count = count +1 
else:
    print(count, "大于或等于5")    # 加else 直接输出五个值

while count < 5:
    print(count, "小于5")
    count = count +1 
print(count, "大于或等于5")     # 与 while 平级 直接输出结果

a = 0
while True:
    print(count, "小于5")
    ount = count +1 
    if count >= 5:
        break                      #  else 也结束
else:
    print(count, "大于或等于5")     # 执行不到 等于没写

a = 0
while True:
    print(count, "小于5")
    ount = count +1 
    if count >= 5:
        break                   #  只是停止while 循环
print(count, "大于或等于5") 

# 循环嵌套      (926 ~~~ 962)
a = 0
while a < 3:                    #  第一次执行 a = 0
    print(a)                    #  第二次执行 a = 1
    a += 1                      #  第三次执行 a = 2
    
    b = 0
    while b < 3:                #  在 a = 0  第一次执行 b 输出 0 1 2
        print(b)                #  在 a = 1  第二次执行 b 输出 0 1 2
        b += 1                  #  在 a = 2  第三次执行 b 输出 0 1 2
"""相当于每执行一次 a 的循环, b 就需要进行一整个循环(也就是输出所有 b )的值"""

a = 0
while a < 2:                    #  第一次执行 a = 0
    print(a)                    #  第二次执行 a = 1
    a += 1                    
    
    b = 0
    while b < 3:                #  在a = 0  第一次执行 b 输出 0 1 2
        print(b)                #  在a = 1  第二次执行 b 输出 0 1 2
        b += 1 

        c = 0
        while c < 4:            #  在b = 0  第一次执行 c 输出 0 1 2 3
            print(c)            #  在b = 1  第二次执行 c 输出 0 1 2 3
            c += 1              #  在b = 2  第三次执行 c 输出 0 1 2 3 
"""每次执行, 先把最后一个(最深层的)执行完, 然后依次往上逐步执行直到结束"""

# 9 * 9 乘法表
right = 1
while right < 10:
    left = 1
    while left <= right:
        print(f"{left} x {right} = {left * right}", end = "\t")
        left += 1
    print()    # 换行, end = "\n"
    right += 1

# for 循环    (964 ~~~ 992)        格式 :   for  变量  in  iterable(可迭代对象 : 除数字外其他的类型)
for i in ["a", "b", (1, 2, 3)]:        # 元素取完就结束
    print(i)        # 输出里面的每一个元素
    print(1)        # 输出元素个数的固定值

for i in range(5):
    print(i)         # 循环五次  原因: range(5) → 0 1 2 3 4 所以 循环五次

str1 = "啦啦啦cccTv1235"
lis1 = [5, 2, 3, "a", "b"]
tup1 = ([1, 2, 3], "a", "b")
dic1 = {"软": "名字", "性格": "沉默"}
set1 = {999, "只要999", "绝不再多一分钱", 888, 777.0}

# 遍历: 进行迭代      整个过程经历一遍
for i in str1:
    print(i)        # 挨个输出元素 不改变顺序

for i in lis1:
    print(i)        # 挨个输出元素 不改变顺序

for i in tup1:
    print(i)        # 挨个输出元素 不改变顺序

for i in dic1:
    print(i)        # 只输出了键 

for i in set1:
    print(i)        # 由集合的无序性输出的结果无序

# for 的条件循环        (994 ~~~ 999)
for i in [1, 2, 3, 4]:
    print(i)        # 3 还是会打印出来, 但是循环不会再进行
    if i > 2:
        break
    print(5)

# for 循环嵌套          (1001 ~~~ 1005)
for i in range(3):
    print(i)                    # 读一次 出一次下面循环
    for j in range(3, 6):
        print(j)                # 3, 4, 5 上面循环几次就出现几次这些数

# 9 * 9 乘法表 for 类          (1007 ~~~ 1011)
for right in range(1, 10):                  # 对应 while 里面 right 取数 并且每次 + 1
    for left in range(1, right):            # 
        print(f"{left}x{right}={left * right}", end = "\t")         # pass 空操作, 被执行时什么都不发生, 临时占位
    print()

# 枚举 enumerate      (1013 ~~~ 1032)
str1 = "abcd"                       # 把可迭代对象中的每个元素和它对应的索引组成一个个元组(index, item). 再把元组构成迭代器
print(list(enumerate(str1)))        # [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')] 索引对应元素
print(dict(enumerate(str1)))        # {0: 'a', 1: 'b', 2: 'c', 3: 'd'} 索引对应元素
for i in enumerate(str1):
    print(i)                        # 同时得到索引和元素, 以元组的形式构成

for index, item in enumerate(str1):         # 分开获得索引和元素                    序列赋值    index, item = (0, "a")
    print(index)                            # 0 1 2 3    
    print(item)                             # a b c d    与上面对应

for index, item in enumerate(str1, start = 100):         # 初始索引从 开始后面的值 100 开始
    print(index)                                         # 100 101 102 103
    print(item)                                          #  a   b   c   d    与上面对应

lis = [1, 2 ,3]
for item in lis:
    lis.append(4)
    lis.pop()
    print(item)                     # 一直循环 下次课见6/9

# break 和 continue         (1034 ~~~ 1048)
# break 终止 [所在! 所在! 所在!] 的循环
for i in range(3):                  # 循环3次, 外面循环三次
    for j in range(3):
        print("hello")              # 里面进行一次直接 break
        break

# continue 跳过当前
a = 0
while a <= 7:
    if a == 3:
        a += 1
        continue                    # 满足条件后, 跳过后面内容  程序内跳过了 3 
    print(a)
    a += 1

# 输出 1~100 所有奇数        (1050 ~~~ 1060)
num = 1
while num <= 100:
    if num % 2 == 0:
        num += 1                    # 对偶数+1 防止变成死循环
        continue
    print(num)
    num += 1                        # 对奇数+1

for i in range(1, 101, 2):
    print(i)

# 求 1~100 之间数的和       (1062 ~~~ 1066)
print(sum(range(1, 101)))           # sum(iterable, start = number(开始数字))

lis = [6, 5, 2, 4]                  
sum(lis, start = 2)                 # start + 6 + 5 + 2 + 4  sum 只能用数字

# 优化石头剪刀布            (1068 ~~~ 1094)
import random

INFO = ("石头", "剪刀", "布")

while True:
    while (player := int(input("请出拳(0 代表石头/1 代表剪刀/2 代表布): "))) not in range(3):           # 判断是否有误 + 输出值
        print("输入有误, 请重新输入: (0, 1, 2)")
    computer = random.randint(0, 2)

    print(f'玩家出拳: {INFO[player]}')
    print(f'电脑出拳: {INFO[computer]}')

    if player == computer:
        print("平局")
    elif player - computer == -1 or player - computer == 2:
        print("玩家获胜")
    else:
        print("电脑获胜")

    while (ans := input("请问您是否需要继续游戏: (Y/N)")) not in ("Y", "N", "y", "n"):           # 判断是否有误
        print("您的输入有误, 请重新输入(Y/N): ")
        # 这里填  continue 也可 ,但是后续判断需要重新定一个 ans

    if ans in ("N", "n"):
        print("游戏结束, 欢迎下次光临!")
        break


"""                                 (1097 ~~~ 1139)
输入一个整数，判断是否为质数        

质数就是一个大于1的自然数, 除了1和它本身外, 不能被其他自然数(质数)整除(2, 3, 5, 7等), 换句话说就是该数除了1和它本身以外不再有其他的因数。
"""
while not (num := input("输入一个整数: ")).isdigit():
    print("你的输入有错误, 需要重新输入: ")            # 得到一个数字

if (num := int(num)) < 2:                            # 判断 < 2 不是质数
    print(f"{num}不是质数")

else:
    for i in range(2, int(num)):
        if int(num) % i == 0:                        # 在这个数前面的数字 不能 整除这个数, 那他就是质数
            print(f"{num}不是质数")                         
            break
    else:
        print(f"{num}是质数")                         # 否则他就为质数

"""
输入一个非负整数 num, 反复将各个位上的数字相加, 直到结果为一位数。

示例:
输入: 38
输出: 2
解释: 各位相加的过程为: 3 + 8 = 11, 1 + 1 = 2; 由于 2 是一位数, 所以返回 2
"""
num = input("请输入一个非负数字: ")                     # 输入一个数!5  1
while int(num) >= 10:
    a = 0                                             # 目的:  重置 a, 以免后面 a 有数导致重复叠加
    for i in num:       
        a += int(i)
        print(a)
    num = str(a)
print(f'得到一位数{num}')

num = input("请输入一个非负数字: ")
while len(num) > 1:
    add = 0
    for i in num: 
        add += int(i)
    num = str(add)
print(num)

# 推导式       先取值 后判断     (1141 ~~~ 1271)
# 列表推导式        (1142 ~~~ 1214)
squares = []
for x in range(10):
    squares.append(x**2)
print(squares)                              # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]  和下面等价
"""相当于"""                                                       
squares = [x ** 2 for x in range(10)]       # 列表推导式 : 结构 一对方括号里面包含一个表达式, 后面跟一个for 子句 然后是 0 个或者多个 for 或 if 子句构成
print(squares)                              # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]  和上面等价


squares = [x ** y for x in range(3, 6) for y in range(3)]       # 嵌套的循环
print(squares)                                                  # [1, 3, 9, 1, 4, 16, 1, 5, 25]
"""相当于    --------------------------------------------         直接执行                              """                                                                       
squares = []                                                    #   👇
for x in range(3, 6):                                           # 3 4 5
    for y in range(3):                                          # 0 1 2
        squares.append(x ** y)                                  # [3**0, 3**1, 3**2, 4**0, 4**1, 4**2, 5**0, 5**1, 5**2]
print(squares)                                                  # [1, 3, 9, 1, 4, 16, 1, 5, 25]


num = input("请输入一个非负数字: ")                   # 各个位数相加
while len(num) > 1:
    add = sum([int(i) for i in num])                # 相当于     add = 0  
    print(add)                                      #           for i in num:                      
    num = str(add)                                  #                add += int(i)
print(num)


squares = [x ** y for x in range(3, 6) for y in range(3, 6)]    # 嵌套的循环
print(squares)                                                  # [27, 81, 243, 64, 256, 1024, 125, 625, 3125]
"""相当于    --------------------------------------------         直接执行                              """                                                                       
squares = []                                                    #   👇
for x in range(3, 6):                                           # 3 4 5
    for y in range(3, 6):                                       # 3 4 5
        squares.append(x ** y)                                  # [3**3, 3**4, 3**5, 4**3, 4**4, 4**5, 5**3, 5**4, 5**5]
print(squares)                                                  # [27, 81, 243, 64, 256, 1024, 125, 625, 3125]


squares = [x ** y for x in range(3, 6) for y in range(3, 6) if x == y ]    # 嵌套的循环 在 if 判断情况下 , 执行
print(squares)                                                  # [27, 256, 3125]
"""相当于    --------------------------------------------         直接执行                              """                                                                       
squares = []                                                    #   👇
for x in range(3, 6):                                           # 3 4 5
    for y in range(3, 6):                                       # 3 4 5 
        if x == y:                                              
            squares.append(x ** y)                              # [3**3, 4**4, 5**5]
print(squares)                                                  # [27, 256, 3125]            


squares = [x ** y for x in range(3, 6) if x == y for y in range(3, 6)]    # 报错, 原因: 在定义y 前用了 y.
print(squares)                                                  # 报错


squares = [x ** y for x in range(3, 6) if x % 2 ==0 for y in range(3, 6)]    # 嵌套的循环 在赋值 x 之后, 进行条件判断, 最后对 y 赋值,
print(squares)                                                  # [64, 256, 1024]
"""相当于    --------------------------------------------         直接执行                              """                                                                       
squares = []                                                    #   👇
for x in range(3, 6):                                           # 3 4 5
    if x % 2 ==0:                                               #   4 
         for y in range(3, 6):                                  # 3 4 5                                            
                squares.append(x ** y)                          # [4**3, 4**4, 4**5]
print(squares)                                                  # [64, 256, 1024]


result  = [(x, y) for x in [1, 2, 3] for y in [1, 2, 3] if x != y]
print(result)                                                   # 把这些数字做成元组放进去
"""相当于    --------------------------------------------         直接执行                              """                                                                       
lis = []                                                        #   👇
for x in [1, 2, 3]:                                             # 1 2 3                                               #   4 
    for y in [1, 2, 3]:                                         # 1 2 3                                            
        if x != y:
            lis.append((x, y))                                  # 把 x 与 y 不等的 组合成元组 依次放进去
print(lis)                                                      # [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]


# 嵌套列表推导式         (1217 ~~~ 1235)
matrix = [[1, 2, 3, 4], 
          [5, 6, 7, 8],
          [9, 10, 11, 12]]      # 隐式的行拼接 参见第二节行结构

result  = [[row[i]] for row in matrix for i in range(4)]
"""相当于"""  
result = []                                     # [[row[0] for row in matrix], [row[1] for row in matrix], [row[2] for row in matrix], [row[3] for row in matrix]]
for i in range(4):
    # result.append([row[i]] for row in matrix)
    # print(result)
    lis1 = []                                    # [row[0] for row in matrix] :[1, 5, 9]    row[1] row[2] row[3]同理 
    for row in matrix:
        lis1.append(row[i])     # 相当于     [row[i]] for row in matrix         

        result.append(lis1)
print(result)                   # 整个内容相当于 [[row[i]] for row in matrix for i in range(4)]

# 字典推导式        (1237 ~~~1253)
result = {x: y for x, y in enumerate("abcd")}
print(result)                       # {0: 'a', 1: 'b', 2: 'c', 3: 'd'} 形成字典
"""相当于""" 
result = {}                 
for x, y in enumerate("abcd"):      # 0-a 1-b 2-c 3-d      
    result[x] = y
print(result)                       # {0: 'a', 1: 'b', 2: 'c', 3: 'd'}

dic = {x: x ** 2 for x in range(4)}
print(dic)                          # {0: 0, 1: 1, 2: 4, 3: 9}

dic = {x: x ** 2 for x in range(6) if x % 2 == 0}
print(dic)                          # {0: 0, 2: 4, 4: 16}

dic = {k: v for k, v in zip((1, 2, 3), (4, 5, 6))}
print(dic)                          # {1: 4, 2: 5, 3: 6}

# 集合推导式        (1255 ~~~ 1271)
set1 = {x ** 2 for x in range(4)}
print(set1)                         # {0, 1, 4, 9}
"""相当于""" 
set1 = set()
for x in range(4):
    set1.add(x ** 2)
print(set1)                         # {0, 1, 4, 9}

set1 = {x for x in "abcferghk" if x not in "abc"}   # 如果x 不是集合中的一个, 就把它加进去 
print(set1)                         # {'k', 'e', 'f', 'h', 'g', 'r'}

set1 = set()
for x in "abcferghk":
    if x not in "abc":
        set1.add(x)
print(set1)                         # {'k', 'e', 'f', 'h', 'g', 'r'}

# 列表内存自动管理      (1273 ~~~ 1295)
lis = [1, 2, 3]                     #  [1  2  3]
for item in lis:                    # 第一次取索引为0的元素 : 移除 1   变成 [2  3]  ;  第二次取索引为1的元素: 移除 3  变成  2
    lis.remove(item)                # 确保元素之间没有间隙
print(lis)
"""
解决办法:   (有新的返回值就行) 原因 : 根据索引删除 
1、创造一份相同的列表, 通过第二个表检索第一个表
2、浅拷贝一份, 根据新的复制列表内容, 删除旧的列表里面的元素 , 深拷贝更可以完成
3、改变数据类型
"""
import copy
lis1 = [1, 2, 3, 4, 5, 6] 
lis2 = [1, 2, 3, 4, 5, 6]
lis2 = lis1.copy()
lis2 = copy.copy(lis1)
lis2 = copy.deepcopy(list1)
lis2 = lis1[::]
lis2 = list(lis1)      # 会返回新的值
lis2 = tuple(lis1)
lis2 = set(lis1)       # 虽然无序, 但是根据要求是删除完里面元素
for item in lis2:
    lis1.remove(item)
print(lis1)

# 遍历问题(只针对字典和集合, 列表不会 列表依据索引) :  改变字典 or 集合大小, 改变原数据个数, 会造成迭代报错
dic = {"name": "Tom", "age": 18, "height": 118}
for i in dic:
    dic.pop(i)
    print(dic)         # 第一次循环没问题, 第二次出了问题;  原因 :   迭代的时候检测到dict 的尺寸改变          

dic = {"name": "Tom", "age": 18, "height": 118}
for i in dic:
    dic.update({"weight": 28})
    print(dic)          # 第一次不报错, 第二次报错 : 原因 : 字典 长度改变, 一改变就报错。
"""
解决办法:   (有新的返回值就行) 原因 : dic2 不变 通过2删1 
1、创造一份相同的列表, 通过第二个表检索第一个表
2、浅拷贝一份, 根据新的复制列表内容, 删除旧的列表里面的元素 , 深拷贝更可以完成
3、改变数据类型
"""
dic1 = {"name": "Tom", "age": 18, "height": 118}     # 还是需要返回新的值
dic2 = {"name": "Tom", "age": 18, "height": 118}        
is2 = lis1.copy()
dic2 = copy.copy(dic1)
dic2 = copy.deepcopy(dic1)
dic2 = dic1[::]
dic2 = list(dic1)      
dic2 = tuple(dic1)
dic2 = set(dic1) 
for i in dic2:
    dic1.pop(i)
print(dic1)              #    {}               

for i in dic1.keys():
    dic1.pop(i)
print(dic1)              # 不可以 , 因为原值改变时, 视图改变------ 参见 141 行


# 函数      (1332 ~~~ 1374)
# 自定义函数        (1333 ~~~ 1359)  

def add(n):              # add 相当于f , n 相当于 x 为形参, print(n + 1) 相当于 x + 1   
    print(n + 1)

add(2)                   # 输出的结果传入 n 的值

add1 = add
add1(6)                  # add1 也指向上面函数, 可以重复使用
"""可传变量"""
num_1 = 4                # num_1 为实参
add1(num_1)             

def trigon():
    for i in range(1, 6, 2):
        print(("*" * i ).center(5))
trigon()                 # 输出一个三角形

def trigon(sign, layers):                            # sign 用来表示的标志,   layers 所需要的最大标志个数
    for i in range(1, 2 * layers, 2):   
        print((sign * i).center(2 * layers - 1))
trigon("!", 6)                                       # 传入实参

def trigon(sign, layers, reverse=False):                   # sign 用来表示的标志, layers 所需要的最大标志个数, reverse 为反转 默认为False
    for i in range(2 * layers-1 , 0, -2) if reverse else range(1, 2*layers, 2):    # 前面 -1 的目的是和后面的2*layers 对应, 左闭右开
        print((sign * i).center(2 * layers - 1))
trigon("*", 5, True)   

# 定义一个和官方一样的函数  (1361 ~~~ 1369)
def my_abs(num):
    if num < 0:
        print(-num)
    else:
        print(num)
my_abs(-3)                 # 没有返回值 或者 返回None  但是官方的有返回值
print(my_abs(-3))          # 3 返回值 None 
print(abs(3))              # 3 返回值 3

a = [1, 2, 3, 4]
for i in a:
    num = add1(i)
    print(num)

# return 的作用      把后面调用的值返回给调用方, 并结束对应的函数           (1376 ~~~ 1451)
def my_abs(num):
    if num < 0:
        return -num
    return num             # 不用写 else return 中 有结束对应的函数 的意义
result = my_abs(-3)        # 流程: 输入3 👉 定义的函数 👉 达到符合的条件 👉 返回结果值
print(result)   

def my_abs(num):
    return -num if num < 0 else num         # 我的问题else 后面加了 return : else 后不加return
print(abs(3)) 

print(my_abs(-3))          # 3 返回值 3 
print(abs(3))              # 3 返回值 3

str1 = "abcd"
print(str1.replace("d", "z"))       # 有返回值 - -- --- 函数里面有 return

lis = ["a", "b", "c", "d"]
lis.append("e")                     # 没有返回值- -- --- 函数里面没有 return
print(lis)

def add(left, right):
    res1 = left + right
    res2 = left * right
    return res1, res2               # 如果为 res1,   会打包成单个元素元组 (res1,)

def add(left, right):
    res = left + right
    return res                      # 调用下面的 , 原因 : 覆盖了

res1, res2 = add(3, 4)              # 解包(序列赋值)
add(3, 4)                           # 会打包成元组

def func():
    print(123141516)
    return                          # 后面不写 相当于返回默认值 None
res = func()
print(res)

def func():
    print(123141516)
print(func())                       # 先输出里面print 内容 再输出返回值
"""无返回值 append print"""

def func():
    for i in range(5):
        print(i)
        return                     # 输出一次后直接返回 None 结束
print(func())                      # 输出 0 None

def func():
    for i in range(5):
        print(i)
    return                         # return 在外, 因此直接结束
print(func())                      # 输出 0 1 2 3 4 None

def func():
    for i in range(5):
        print(i)
        break                      # 打断for 循环直接默认 return 返回 None  return 在外面
print(func())                      # 输出 0 None

def func():
    for i in range(5):
        print(i)
        return                     # 打断循环 直接输出返回内容 下面不输出
    print("ending...")
print(func())                      # 输出 0 None

def func():
    for i in range(5):
        print(i)
        break                      # 打断for 循环直接默认 return 返回 None  return 在外面
    print("ending...")             # 所以这一步会输出
print(func())                      # 输出 0 None ending...

# 文档注释       (1453 ~~~ 1453)
def my_abs(num):
    """
    Return the absolute value of the argument.
    """
    return  -num if num < 0 else num
my_abs(-123)

print(my_abs(-6))
print(my_abs.__doc__)       # 查看文档注释

# help的使用            在终端输入内容: 可以查看代码   退出按Q

# 常用函数      (1466 ~~~ 1492)
"""
1、abs 返回绝对值
2、divmod(a, b)   a > 0  b > 0   返回一个包含商和余数的元组 (a//b, a % b)
3、max(iterable/ [iterable, ])   找出并返回最大值
4、min(iterable/ [iterable, ])   找出并返回最小值
5、pow(base, exp, mod(可不写))    返回 base 的exp 次幂 前面两个 base, exp 为必须参数; mod 为返回 base 的exp 次幂后 取余
6、round(num, ndigits)           ndigits 小数点位数, 
"""
# 上面的 4  ----- min
print(min([1, 2, 3]))       # 非空可迭代对象(iterable), 返回 返回 返回! 其中的最大值, 输出结果
print(min((1, 2, 3, 4)))    # 传入多个数值也可, 输出4
print(min("1", "2", "3", "10"))      # "3"   字符串比较时 按照 拿出第一位来比较 , "10" 的第一位 是 1  所以小于 "3"
print(min([], default = 666))        # 传入空的 iterable 前面为空的时候, 必须要有一个默认值
print(min([1, 2, -3], key = abs))     # key 与 sort 或 sorted 的情况相同,  指定函数  

# 上面的 5
pow(-2, 3, 5)                # 2 的 3次方后对 5 取余数  结果为 -2   -2**2 % 5

# 上面的 6
print(round(0, 3))           # 不是向下取整, 距离前后整数相同时, 取整为偶数 ; 小数点后为0 直接输出整数

print(round(-0.5, 0))        # ndigits 不为 None 返回的值和 num 的类型相同  被省略或者为None 则返回整数  

print(round(-1.5, 3))        # 先判断前面 num 和哪个偶数接近, 之后保留小数

print(round(-1.5, 0))        # 选偶数 ,只有不写或者为 None 为整数 其他会有 x.0 

# 类型标注      (1493 ~~~ 1500)
def func(a: str, b: str, c: list, d: tuple, e: dict, f: set):
    print(a, b, c, d, e, f)
func(123)

print(func(1, 1, 1, 1, 1, 1))

print(func.__annotations__)

# 参数传递      (1502 ~~~ 1528)
def func(b):
    print(id(a), a)             # a 和 b 是同一个 999  id相同
    print(id(b), b)
    b = 888
    print(id(a), a)             # a 和 b 不是同一个 888  b 的新值覆盖了b
    print(id(b), b)
a = 999
func(a)

def func(b):
    print(id(a), a)             # a 和 b 是同一个 [999, 777] id相同
    print(id(b), b)
    b = [999, 777]
    print(id(a), a)             # a 和 b 不是同一个 [999, 777]  b 和 a 的地址不一样
    print(id(b), b)
a = [999, 777]
func(a)

def func(b):
    print(id(a), a)             # a 和 b 是同一个 [999, 777] id相同
    print(id(b), b)
    b.insert(1, 888)
    print(id(a), a)             # a 和 b 是同一个[999, 888, 777]  直接引用过去, 对原数据进行修改, 整个会发生变化, print(a) 也为[999, 888, 777]
    print(id(b), b)
a = [999, 777]
func(a)

# 参数分类      (1530 ~~~ 1581)
# 必须参数 : 必须要传 不传报错
def func(a, b):
    print(a - b)
func(4, 3) 

# 关键字参数:    实参用  ----- 通过关键字传入值
def func(name, age):
    print('姓名: ', name)
    print('年龄: ', age)
func("火之高兴", 50)

# 默认参数 不能写在必须参数前 在形参用 
def func(name, age=60):                     # 会直接赋值
    print('姓名: ', name)
    print('年龄: ', age)
func("火之高兴")

# 不定长参数  参数前单个星号(*args) 打包成元组, 如果没值, 则返回空元组 ; 参数前带两个星号(**kwargs) 打包成字典, 如果没值, 返回空字典;
def func(a, b, *args):                      # 先把必须参数喂饱, 剩下给不定长参数
    print(a, b, args)
func(1, 2)                                  # 1 2 ()

def func(a, b, *args):                      # 先把必须参数喂饱, 剩下给不定长参数
    print(a, b, args)
func(1, 2, 3, 4)                            # 1 2 (3, 4)

def func(a, b, **kwargs):
    print(a, b, kwargs)
func(1, 2, T=3, F=4)                        # 只能接受关键字参数 是一个键值对
 
# 参数顺序  必须参数>>单星号参数*args <=可交换=> 默认参数c >>双星号参数**kwargs
def func(a, b, *args, **kwargs):
    print(a, b, args, kwargs)
func(1, 2, 3, 4, 5, 6, F=3, t=4)            # 1 2 (3, 4, 5, 6) {'F': 3, 't': 4}

def func(a, b, *args, c=9, **kwargs):       # 有值优先用值 元组已经打包 所以 c = 9  (3, 4, 5)打包成元组
    print(a, b, args, kwargs, c)
func(1, 2, 3, 4, 5, 6, F=3, t=4) 

def func(a, b, c=9, *args, **kwargs):       # 有值优先用值 元组已经打包 所以 c = 3   (4, 5) 打包成元组
    print(a, b, args, kwargs, c)
func(1, 2, 3, 4, 5, 6, F=3, t=4) 

# 特殊参数
def func(a, /, b):    # / 之前的的所有的参数, 传实参的时候必须以位置参数的形式传入, 
    print(a - b)
func(4, 2)            # 例如 a = 3 会报错

def func(a, *, b):    # * 之后的的所有的参数, 传实参的时候必须以关键字的形式传入, 
    print(a - b)
func(4, b=2)          # 限制后面必须要 b=int() 输入

# 匿名函数  (1583 ~~~ 1637)     不需要return 可以在任意情况使用
lambda canshu, canshu2, canshu3 : expression
func1 = lambda : "it just returns a string"
print(func1())

a = lambda : print(123456)   # 定义了一个匿名函数
print(a())                   # 123456 输出的数 None 返回值 , 原因 print 没有返回值

a = lambda : (123456, 6)     # 定义了一个匿名函数
print(a())                   # (123456, 6) 返回值

a = lambda : print(123456); 23; print(123456, 6)    # lambda 后面只能接一个表达式
a()                # 123456 输出的数 None 返回值 , 原因 加了 ; 后相当于后面的数据类型是新的一行

func2 = lambda x, y, z :  x + y + z
print(func2(1, 2 ,3))            # 返回值是 表达式的结果 (上面的式子)

from typing import Callable, Iterator      # Callable  可调用参数
def call_f1(function: Callable):
    print(function)
func1 = lambda : "it just returns a string"
call_f1(func1)                                      # 指向了他的地址, 只是地址和下面不一样  原因 : 创建了一个func1 的新地址
call_f1(lambda : "it just returns a string")        # 指向了他的地址, 只是地址送上面不一样 
"""相当于上面"""
def func1():
    return  "it just returns a string"

def call_f2(f, a, b, c):
    return f(a, b, c)                       # 执行 f 把 a, b, c 作为参数

func2 = lambda x, y, z : x + y + z          # 相当于 func2 = x + y + z 然后输入参数 传到函数中, 之后输出结果 灵活性强 功能不强
call_f2(func2, 2, 3, 4)
print(call_f2(func2, 2, 3, 4))

func2 = lambda x, y, *z : print(x, y, z)         # 前面加 * 还是把剩下数字变成元组 不能进行建议
def func2(x, y: int, *z):                        # 前面加 * 还是把剩下数字变成元组 可以进行建议
    print(x, y, z)
func2(1, 2, 3, 4)

func2 = lambda x, y, z: x + y + z           # 相当于 func2 = x + y + z 然后输入参数 传到函数中
func3 = lambda f, x, y, z: f(x, y, z)       # 相当于 func3中包含一个函数 f
func3(func2, 2, 3, 4)                       # 函数中包含 func2 在定义完 func2 之后 在func3 里面传入func2 x y z

lis = ["1.abc", "100.a", "2.b", "8.cb"]
lis.sort(key=lambda string: int(string.partition(".")[0]))
print(lis)  
"""等同于上面"""
lis = ["1.abc", "100.a", "2.b", "8.c"]
lis.sort(key=lambda x : int(x.split(".")[0]))
print(lis)

# 文件名从小到大排序:  FileName = ["10.py", "2.py", "8.py", "6.py", "100.py"]
FileName = ["10.py", "2.py", "8.py", "6.py", "100.py"]
FileName.sort(key=lambda x : int(x.split(".")[0]))
print(FileName) 

# 解包      (1639 ~~~ 1684)
a, b, c, d  = 1, 2, 3, 4
print(a, b, c, d)

a, b, c, d  = "abcd"
print(a, b, c, d)

a, b, *c, d, e = "abde"      # 带 星号 代表可以对应 0 个元素 , 没有元素变成空列表   不允许带两个星号
print(a, b, c, d, e)         # a b [] c d

a, b, *c, d, e = "abcde"     # 带 星号 代表可以对应 0 个元素 , 没有元素变成空列表
print(a, b, c, d, e)         # a b ['c'] d e 打包成列表

a, b, *c, d, e = "abcdefghjkl"     # 带 星号 代表可以对应 0 个元素 , 没有元素变成空列表
print(a, b, c, d, e)               # a b ['c', 'd', 'e', 'f', 'g', 'h', 'j'] k l  对应上面的 , 对于多的元素会打包成列表, 左边对应左边, 右边对应右边

*c, = "abcdefgh"
print(c)                     # 打包成列表相当于 存在 c = "abcdefgh"  list(c)
list(c)

lis = [1, 2, 3, 4, 5, 6]
_, *b, _ = lis              # 变量用不上的时候用下划线
print(b)                    # [2, 3, 4, 5]

a = (1, 2, 3, 4, 5)
print(*a)   # 1 2 3 4 5
print(* (1, 2, 3, 4, 5))    # 同上

dic = {"a": 1, "b": 2, "c": 3}

*a, b, c = 1, 2 ,3 ,4 ,5    # 打包列表 [1, 2 ,3] 4 5
print(*a)                   # 对a 进行解包, 变为 1 2 3 

def func(a, b, c, d=None):
    print(a, b, c, d)
tup = (1, 2 ,3 ,4)
dic = {"name": "张三", "age": 18, "hight": 178}
func(*tup)      # 变成普通数字
func(*dic)      # 解开iterable 没有的参数为 d 的固定值  

# 双星号 处理对象是字典
def func(a, b, c, d=None):
    print(a, b, c, d)
dic = {"a": 1, "b": 2, "c": 3}
func(**dic)
func(a= "TOM", b=18, c=True)        # 在实参前面打两个** 把实参解开 必须键值对相符 b 改为 其他则报错

# 命名空间  从一个名称到对象的映射   实现 : 大部分由字典实现    内置命名空间由:builtins     (1686 ~~~1726)
# 作用 :  避免命名冲突      
def func():
    print(str(123))
    
    def func2():
        str = 123
        print(str(123))
    return func2

import builtins             # dir 返回当前范围内变量, 方法和定义的类型列表              
print(dir(builtins))

# 全局命名空间 : 整个文件里的的外皮内容
# 局部命名空间 : 记录了函数的变量参数  调用函数, 会创建局部变量空间
# 内置命名空间 包含 全局命名空间 和 局部命名空间 

def func1(arg1, arg2):
    num = 666                # 如果函数里面不传 则使用全局变量
    print("func1-globals: \n", globals(), end = "\n\n")         # 返回当前全局命名空间的字典
    print("func1-locals: \n", locals(), end = "\n\n")           # 返回当前局部命名空间的字典
    return num, arg1, arg2

def func2(arg1, arg2):
    num = 777
    print("func1-globals: \n", globals(), end = "\n\n")         # 返回整个文件全局命名空间的字典    全局变量记录了 python文件 ,导入包的地址, 函数 以及外部的 num = 111 在函数外都是全局
    print("func1-locals: \n", locals(), end = "\n\n")           # 返回当前局部命名空间的字典    局部变量记录了 不同空间(函数)存放的 num  在函数内部
    return num, arg1, arg2

num = 111
func1(222, 333)
func2(444, 555)

# 在全局命名空间下, globals () 和 locals() 返回相同的字典
print(globals())
print(locals())

# 命名空间是个容器 ,存放各种名称   有多个命名空间存在一样名称 它们是互相独立的 , 检索方式: 先检索局部 再检索全局 再检索内置命名空间 找不到报错 : 没有定义
str = "ab"
lis = list(str)
print(str(lis))                     # 内置命名空间有 str ; 全局也有 str; list 为 内置命名空间的list 

"""  (1728 ~~~ 1788)
作用域  
定义 : python 可以直接访问命名空间的正文区域
作用 : 决定了哪一部分区域可以访问哪个特定的名称
分类 : (L-E-G-B 作用域逐渐增大)   局部作用域 -- 闭包函数外的函数 -- 全局作用域 -- 内建作用域
"""
# python 程序可以直接访问命名空间的正文区域
def func(arg1: int) -> int:     
    a = 2                       # 局部命名空间 存放这个 a  
    return arg1 + a             # 与上面 a = 2 为作用域,    访问 {arg : 1} 访问不到外面

a = 1                           # 全局命名空间 存放这个 a  为作用域 访问 {func: 1} 
print(func(a))

def func():         # 优先在局部 之后 全局 再之后 内置 在找不到 incloding  最后找不到 报错
    a = 2           # 局部变量
    b = 3           # 局部变量
    print(a + b)    # 局部作用域可以调用局部变量
    print(d)        # 局部作用域可以调用全局变量
d = 4
a = 100
func()              # 全局变量不能调动局部变量

print(a, b)         # 报错, 全局变量不能调用局部变量

# 闭包函数外的函数中    (1753 ~~~ 1779)
def outer():    # 单步调试 直接跳到第十五行,只是定义了没有调用        | 作用过程 def outer 👉 a = 1 👉 outer 👉 outer 内的作用域
    b = 2       # Enclosing 变量b,                                 |
    c = a + 3   # Enclosing可以调用全局变量a  此处的c 没在全程调用   |                  👉 def inner 👉 return inner 👉执行inner c = 5 这里和上面 c = 4 不冲突 地址不一样
                #                                                 |                  👉  print的内容               
    def inner():    
        c = 5       # 局部变量c
        print(a)    # 局部作用域调用全局变量
        print(b)    # 局部作用域调用Enclosing 变量 b
        print(c)    # 优先调用本局部变量c
    return inner()
a = 1 # 全局变量
outer()

def outer():
    a = c + 2           # 调用全局c
    c = 98              # 此处的c 不会执行

    def inner():
        b = c + 2       # 调用的 c 有歧义
        print(a + b)    # 调用上面的 a

    return inner()

c = 1           # 全局变量
outer()
print(c)        # 调用全局变量c

# 内建作用域  只有类 函数 模块 才能生成新的作用域  (1781 ~~~ 1788)
def func(string, *index):
    s = ''
    for i in index:
        s += string[start: i]
        start = i + 1
    s += string[start: ]
    return s      

# global 和 local               (1809 ~~~ 1909)
def func():
    b = a + 1                   
    print(b)
a = 4                           # 全局变量
func()

def func():
    a += 1                      # 这个 a 为局部变量  等号左边的是局部变量 右边的是全局变量(理论上来说)                 
    print(a)                    
a = 2   
func()                          # 报错 , 局部变量a 在赋值前使用   不符合逻辑

def func():
    a = 4                       # 局部作用域内                                                   
a = 2                           # 全局作用域内 
func()
print(a)                        # 2    优先自己区域找 , 这个a为全局 哪怕外面没有也不会调用里面的 a

def func():
    global a                    # 声明: 声明后面的 a 都是全局变量的 a, 之后的 a 全为 全局的a
    a = 4                       # 局部作用域内                                                   
a = 2                           # 全局作用域内 
func()
print(a)                        # 4    原因: 被声明为全局, 相当于对 a 重新做了赋值

def func():
    global a
    a = 4                       # 局部作用域内
    a = a + 1                   # 声明: 声明后面的 a 都是全局变量的 a, 之后的 a 全为 全局的a                                                                       
a = 2                           # 全局作用域内 
func()
print(a)                        # 5    原因: a = 4 已经被声明为全局变量 

def func():  
    a = 3                       # enclosing 作用域
    def func2():
        # global a              # 输出 6 调用全局变量(外面的)
        nonlocal a              # 输出 4 调用局部变量(enclosing内的)  声明后文的 a 都为 enclosing(闭包函数外的函数)作用域的 a
        b = a + 1
        a = a + 1
        print(a, b)
    return func2()
a = 5
func()                          # 结果为 4 4 在 a= 3 的时候 对 a 重新赋值
"""
总结 : 里面的没有那个变量会找外层的, 外层的没有那个变量 只有在内层加了 global 后才能找内层的 ; 
global 直接调用全局的 , 可以在函数(局部作用域)内改变数据, 
nonlocal 优先调用函数内的, 也就是在 enclosing 作用域内的变量 他不会调用全局的变量 如果encoding 作用域没有会报错
"""

def func():
    global a                
    a = 3                      # global 后 a 重新赋值为 3 
    def func2():
        global a               # 因为执行了函数func2 所以下面的 a 也变成全局变量a
        # nonlocal a           # 在global 3 的条件下 总体输出16 这个内容报错, 会影响到 a 的重新赋值    name 'func' is not defined
        a = a + 1              # 再次重新赋值, 全局变量a 变成了 4 
    return func2()             # 相当于执行了函数 func2
a = 16
func()
print(a)                       # 输出 4  原因 :  global 后 a 重新赋值为 3 之后在里面调用函数, 变成 4

def func():
    global a                
    a = 3                      # global 后 a 重新赋值为 3 
    def func2():               # 没有执行, 所以后面不运行
        global a                  
        # nonlocal a           # 在global 3 的条件下 总体输出16, 在同样无法调用函数 func() 会影响到 a 的重新赋值    name 'func' is not defined
        a = a + 1              
    return func2               # 此条没有调用函数 func2 因此 没有执行
a = 16
func()                         # 这个结果为func2  只是一个函数的地址
func()()                       # 这个结果等价于func2() 试验用, 不要运行计入下面结果 这个输出结果为 4 
print(a)                       # 输出结果为 3 原因 只进行了在global 后 a 的重新赋值                   

def outer():
    global a, b                # 声明当前作用域的a,b为全局变量
    a, b, c, d = 111, 122, 123, 6
    print(a, b)                # a 重新赋值 111  b 重新赋值 122                           

    def inner():
        global a, b            # 声明当前作用域的a,b为全局变量
        nonlocal c, d          # 声明当前作用域的c,d为Enclosing变 量  这里的 c d 为上面的 123 ,6 
        a, b, c, d = 11, 22, 23, 666
    inner()   
    print(c, d)                # c重新赋值 23  d 重新赋值 666
a, b = 3, 4 
outer() 
print(a, b)                    # 11 22  原因: 赋值两次 最后赋值到 inner 内的 11 , 22

def outer():
    lis = [1, 2]

    def inner():
        nonlocal lis           # 不加会报错 原因: 上面的是局部变量, 不会在这里使用
        res = lis.append(3)
        lis = lis.append(3)    # 通过这一步变成了把上面的lis 变成了None
        print(lis, res)        # None None 原因: append 无返回值 
    return inner()
outer()

def func():
    res = [1, 2, 3, 4]          # 定义一个res
    num = 4                     # 这里的 4 不会被用到

    def pop_num():
        global num
        nonlocal res            # 用的 res 的列表
        res.pop(num)            # pop 会返回res 
        res1 = res.pop((1))     
        res2 = res.append((4, 5))
        print(res1)             # 2  原因 : 新定义 res1 结果需要依据所用的函数是不是存在返回值 来进行判断是否为None pop 有返回值 所以 输出2
        print(res2)             # None    原因 :  append 无返回值

    return pop_num()            # return 的是func 里面的内容

num = 3
func()
print(func())                   # None

# 高阶函数    定义 : 参数或(和) 返回值为其他函数的函数        (1911 ~~~ 1984)
"""以前学过的 sorted, max, min, """
# filter(function, iterable)    function 中的必需参数只能有一个(取决于功能) 也可以为None  
"作用: 根据函数结果返回 判断是否为False 或者 True 将判断为True 的iterable 中的元素构建新的迭代器并返回  "

# abs(0), abs(-2), abs(-5), abs(True), abs(3), abs(False)  => 0 2 5 1 3 0
# 根据结果进行bool判断: False True True True True False 映射到原数据上
# 把结果为True的元素全部拿出来, 构建一个新的迭代器
print(list(filter(abs, [0, -2, -5, True, 3, False])))    # [-2, -5, True, 3]

# temp = lambda i: abs(i)-1
# 判断 abs(-1), abs(1), abs(False), abs(2)   => 0 0 -1 1
# 进行bool判断 False, False, True, True 映射到原数据上
# 把为True 的拿出来构成一个新的迭代器
print(list(filter(lambda i: abs(i)-1, {-1: 3, 1: 4, False: True, 2: 5})))            # 字典会以字典的key 来进行参与

# 因为第一个参数为None 即直接对可迭代对象的元素进行 bool 判断:  True True False True 映射到原数据上
# 把结果为True的元素拿出来, 即 False 2 构成一个新迭代器
print(list(filter(None, {-1: 3, 1: 4, False: True, 2: 5})))     # [-1, 1, 2] 原因 : 因为为None 所以直接判断内容


# temp = lambda i: print(abs(i)-1)
# temp(-1), temp(1), temp(False), temp(2)  => 输出 0 0 -1 1 并返回 None None None None
# 根据结果进行bool 判断 : False False False False 映射到原数据上
# 将结果为True的元素拿出来, 没有True,构成一个新的迭代器
print(list(filter(lambda i: print(abs(i)-1), {-1: 3, 1: 4, False: True, 2: 5})))     # 输出为 []

# map(func, *iterables) 映射  func 必须和可迭代对象的个数相同  相当于给内置函数传值
print(list(map(lambda x: x**2, (1, 2, 3, 4))))          # [1, 4, 9, 16]
print(list(map(lambda x, y: x + y, (1, 2, 3, 4), (5, 6, 7, 8))))    # (1, 2, 3, 4) 对应 x   (5, 6, 7, 8)对应 y
print(list(map(pow, (1, 2, 3, 4), (5, 6, 7, 8), (1, 2, 3, 4))))     # 看pow的参数的值, 返回的是x的y次方的值

res = input("请输入一串数字用逗号隔开: ")   # "1, 2, 3, 5, 789"
lis = res.split(",")
lis1 = [int(i) for i in lis]
lis2 = list(map(int, lis))                # 里面每个数字都转为 int 类型

# square(1), square(2), square(3) => 1 4 9  并返回 [None, None, None]
def square(a):
    print(a ** 2)
result = map(square, [1, 2, 3])
print(list(result))                       # [None, None, None]

# reduce(function, iterable, initial(初始值))
"""
需要调用 functools 模块
function 必须参数只能有 2 个
在没有指定 initial 参数时，先把 iterable 的前两个元素作为参数调用函数, 把这次函数的结果以及iterable 的下一个元素又作为参数再调用函数  类似于递推
"""
from functools import reduce
def add(m, n):
    s = m + n
    return s
# 先从 iterable 中取前两个元素作为add 调用, 结果为 add(1, 2)  => 3
# 再把上一步得到的结果作为add 的参数调用, 结果为add(3, 3)  => 6
# 再把上一步得到的结果作为add 的参数调用, 结果为add(6, 4)  => 10
result = reduce(mul, [1, 2, 3, 4])          # 1* 2 *3 *4
print(result)

# mul(-2, 1)  => -2
# mul(-2, 2)  => -4
# mul(-4, 3)  => -12
# mul(-12, 4)  => -48
# mul(mul(mul(mul(-2, 1), 2), 3), 4) 
result = reduce(mul, [1, 2, 3, 4], -2)      
print(result)

# 如果可迭代对象为空 值接返回初始值, 没有初始值则报错
result = reduce(lambda x, y: 10*x + 2*y, [], 123)       # 只指定了初始值   返回这个初始值
print(result)       # 123

# 如果有一个元素, 未指定初始值, 直接返回这个元素
result = reduce(lambda x, y: 10*x + 2*y, [1])           # 只指定了元素 返回这个元素
print(result)       # [1]


# 实现最大公约数算法（只考虑正整数）
# 实现最小公倍数算法（只考虑正整数）

def num_gcd(a, b):
    for i in range(min(abs(a), abs(b)), 0, -1):
        if not a % i and not b % i:                     # 啥都不写相当于 a % i == 0
            return f'{a}, {b}的最大公约数是{i}'
    return a if a  else b                               # 这里的 if 后的 a 为 a == 0

print(num_gcd(0, -9))

# 递归问题              (1998 ~~~ 2026)
# 兔子问题, 一只兔子每两个月成年才能产子 n 个月后能有多少只兔子
def get_rabbit(n):
    if n < 2:
        return 1
    return get_rabbit(n-1) + get_rabbit(n-2)
print(get_rabbit(3))

def get_rabbit(n):
    n0 = 1
    n1 = 1
    for _ in range(n-1):    
        n0, n1 = n1 , n0 + n1
    return n0

import sys
print(sys.getrecursionlimit())          # 最大递归深度为1000
sys.setrecursionlimit(1500)             # 设置最大递归深度为1500

dic = {}                                            # 定义一个字典
def get_rabbit(m):
    if m < 2:
        return 1
    if m in dic:                                    # 如果m 在这个字典中, 返回m 对应的这个值
        return dic[m]
    result =  get_rabbit(m-1) + get_rabbit(m-2)     
    dic[m] = result                                 # 给字典赋值
    return result                                   # 如果不成立, 则返回result 再次计算 
print(get_rabbit(998))

# 如何选出质数 (2028 ~~~ 2036)
while not (num := input('请输入一个数: ')).isdigit():
    print('您输入的有误, 请重新输入一个整数')
for i in range(2, int(num)):            # 在 2 ~ 这个数字内是否会被里面的数字整除, 如果会则这个数不是质数
    if int(num) % i == 0:
        print(f'{num}不是质数')
        break
else:
    print(f'{num}是质数')



# 面向对象部分
# 面向过程(结构化编程)      (2041 ~~~ 2078)
name = '张三'
age = 18
adres = "上海市"
counter = 0
print(f'我叫{name}, 我今年{age}岁, 我现在在{adres}, 等待疫情结束.')
counter += 1
print(f'{name}起床了')

def morning(name):
    print(f'{name}刷牙')
    print(f'{name}洗脸')
    print(f'{name}吃早餐')
morning(name)

def middle_morning(name):
    print(f'{name}开机')
    print(f'{name}上课')
middle_morning(name)

def midday(name):
    print(f'{name}吃午饭')
    print(f'{name}午休')
    print(f'{name}仰卧起坐')
midday(name)

def afternoon(name):
    print(f'{name}码代码')
    print(f'{name}写作业')
    print(f'{name}出门跑步')
    print(f'{name}吃下午饭')
afternoon(name)

def night(name):
    print(f'{name}看回放')
    print(f'{name}看番剧')
    print(f'{name}沉眠')
night(name)

# 面向对象(高内聚低耦合)   (2080 ~~~ 2168) 程序 =  数据 + 算法
"""
class mydaily(object):   object 是所有类的父类
所有的类都默认继承object 类, object 就是所有类的基类(父类), mydaily则为子类
父类里面存在__new__  可以不用写, 
父类里面有 __init__ 为什么要写, 原因: 需要定制化;  存在的原因 :  万一某天不需要定制化, 可以直接拿来 
"""
class Mydaily():                                                      # 这个是类对象, 类, 类对象; self 指的是实例对象 代表后面定制内容所代表的传入变量
    def __init__(self, name, age, adres):                             # 初始化方法(其实是一个魔术方法), 用来初始化一个事物的属性; 初始化方法: 1、self; 2、其他参数
        self.name = name                                              # md[i] 的名字叫XXXX     实例变量(属性)
        self.ages = age                                               # md[i] 的年龄是XXXX     实例变量(属性)
        self.adres = adres                                            # md[i] 的地址是XXXX     实例变量(属性)

    def __new__(cls):                                                 # 为什么没有调用  原因 : 因为继承, 所有的类都默认继承object 类, object 就是所有类的基类(父类), daily则为子类
        pass

"""
构造方法(其实是一种魔术方法)
先调用, __new__(cls), 然后把类名传给cls参数, 构造出一个实例对象, 最后返回
__new__构造出的实例对象在返回之前, 还要把这个构造出来的实例对象(丢给self)丢给__init__(self)进行定制(name, age, adres, 其他属性)也丢给上面默认参数
最后才把定制好的实例对象通过__new__返回

下面四次定制不用, 返回了四个值对象
"""
md1 = Mydaily("沐木", 22, "上海市杨浦区")                              # 实例化, 返回一个实例对象 人啊 物啊 地点啊  会把三个实参甩给上面默认函数定义三个变量
md2 = Mydaily("艾冰", 24, "上海市闵行区")                              # 在传入后self 的作用就是 告诉你有几个对象
md3 = Mydaily("封橘", 26, "上海市浦东新区")                             
md4 = Mydaily("虹彩", 23, "上海市杨浦区")

print(md2.name, md2.ages, md2.adres)                                 # 实例变量是每个实例对象所独有的  在实例化的过程中 self 不同 所存储的地方不同
"""
初始化方法和构造方法的区别 :
在实例化的过程当中, 实际上是初始化方法和构造方法共同完成一个实例对象的构建, 实例对象由__new__(cls)来构建, 由__init__(self)来定制;
例如一个人画设计图建造, 另一个人来美化, 所以__new__为构造方法, __init__(self)为初始化方法, 进行定制(属性),
__init__(self)进行定制(属性)只是单纯的进行定制, 最后成品返回是由__new__构造方法返回, 基于这样的原因, 所以__init__不能有返回值, 即返回None    
"""

# 如何定义类属性  通常定义类, 首字母需要大写
class Mydaily():                                                      # 这个是类对象, 类, 类对象; self 指的是实例对象 代表后面定制内容所代表的传入变量

    school = '交大深兰'                                               # 类变量(属性)  是大家公有的, 所以实例对象可以调用他

    def __init__(self, name, age, adres):                             # 初始化方法(其实是一个魔术方法), 用来初始化一个事物的属性; 初始化方法: 1、self; 2、其他参数
        self.name = name                                              # md[i] 的名字叫XXXX     实例变量(属性)
        self.ages = age                                               # md[i] 的年龄是XXXX     实例变量(属性)
        self.adres = adres


md1 = Mydaily("一号人员", 22, "上海市杨浦区")                         # 实例化, 返回一个实例对象 人啊 物啊 地点啊  会把三个实参甩给上面默认函数定义三个变量
md2 = Mydaily("二号人员", 24, "上海市闵行区")                          # 在传入后self 的作用就是 告诉你有几个对象
md3 = Mydaily("三号人员", 26, "上海市浦东新区")                             
md4 = Mydaily("四号人员", 23, "上海市杨浦区")

""" 实例属性(变量)只能由实例对象调用, 是独立的, 实例变量是每个实例对象所独有的; 类不能单独调用实例变量(属性) """
print(md2.name, md2.ages, md2.adres)    
print(Mydaily.name)                                                  # 报错, 属性错误没有 name 属性

""" 类方法调用规则: 推荐用类去调用, 尽管实例对象也能调用 """
print(Mydaily.school)
print(md1.school)                                                               

# 如果单独给一个实例对象加一个参数 动态定义变量
"""
动态定义实例变量: 如果实例属性存在则修改, 如果不存在则定义变量.
实例变量是实例对象所独有的.
"""
md2.height = 181
print(md2.height)           # 只给这一个加, 只是在单个实例对象所独有的, 其他人不加
print(md1.height)           # 报错, 没有height 这个变量 原因: 实例变量是实例对象所独有的 只是单独给2加了

md1.ages = 21               # 可直接修改变量
print(md1.ages)

""" 动态定义类变量: 如果类属性存在, 则修改; 不存在, 则定义新的类变量 """
Mydaily.school = "深兰, 交大"       # 改变了原值
print(Mydaily.school)

Mydaily.buy = "日常购物"            # 定义一个新的类变量
print(Mydaily.buy)

""" 类变量是所有实例对象公有的 """
print(md1.buy, md2.buy, md3.buy, md4.buy)                     # 实例对象也能引用类变量修改过的

"""
是否可以用实例对象定义一个类变量: 不行 原因 👉 实例对象所定义的内容只是在动态定义实例变量, 不能定义类变量;
实例对象调用类变量重新赋值, 其实是在动态定义变量, 根本与类变量无关 
"""
md1.buy = "某件衣服"
print(md1.buy)                  # 某件衣服
print(md2.buy)                  # 日常购物
print(Mydaily.buy)              # 日常购物

# 方法(类方法, 对象方法, 静态方法)       (2172 ~~~ 2321)
"""
区分函数和方法: 在类里面的就是方法, 外面就是函数
方法通常有参数，比如对象方法隐式的接收了 self 参数，
类方法隐式的接收了 class(cls) 参数
函数可以没有参数
"""
class rush():
    def __init__(self, name, gard, weapon):
        self.name = name
        self.gard = gard
        self.weapon = weapon

    def kill_boss(self):                                 # 加了 self 能够调用到 self 对应的属性和方法; 同时消除作用域                                 
        goal = "杀死食人魔首领"
        adres = "食人魔山"
        return f'{self.name, self.gard}接受了{goal}的任务, 拿着{self.weapon}, 正在前往{adres}'
    # 与上面相比没有self参数, 下面的无法引用实例对象, 因为作用域不一样 下面报错
    # def kill_boss():                    
    #     goal = "杀死食人魔首领"
    #     adres = "食人魔山"
    #     return f'{self.name, self.gard}接受了{self.goal}拿着{self.weapon}正在前往{self.adres}'

# 对象方法调用 
p1 = rush("法师", 60, "法杖")
p2 = rush("战士", 57, "剑盾")
p3 = rush("猎人", 60, "弓弩")
p4 = rush("术士", 44, "法杖")
""" 方法调用很像, 调用 kill_boss 的时候没有传参数, 为什么不需要传实参 : 因为 实例对象 在调用 对象方法 时有传参数(self), 而调用对象不同, 所以self不同 """
p1.kill_boss()                          
p2.kill_boss()
p3.kill_boss()
p4.kill_boss()
# 下面四个等同于上面四个
rush.kill_boss(p1)
rush.kill_boss(p2)
rush.kill_boss(p3)
rush.kill_boss(p4)

"""
对象方法是否可以用类调用: 不能
所以对象方法只能由实例对象去调用
"""

# 定义一个类方法
class rush():

    bases, Npc = "主城", "某NPC"

    def __init__(self, name, gard, weapon):
        self.name = name
        self.gard = gard
        self.weapon = weapon

    def kill_boss(self):    # 对象方法  第一个需要传入对象, 后面可以加其他参数; 这里self可改变, self 对应实例对象;  加了 self 能够调用到 self 对应的属性和方法; 同时消除作用域                                 
        goal = "杀死食人魔首领"
        adres = "食人魔山"
        return f'{self.name, self.gard}接受了{goal}的任务, 拿着{self.weapon}, 正在前往{adres}'

    @classmethod    # 类方法装饰器      第一个参数需要传入类, 后可接其他参数                                  
    def finish_goal(cls, num):                                          # cls 对应类对象, 后面也可以加其他参数, 实参就是前面的rush(这个类) , 这里cls 可以改为其他名字
        print(f'{num}完成了任务, 返回{cls.bases}寻找{rush.Npc}交任务')    # 这里cls 和 rush 是一个东西
        
    # 静态方法
    @staticmethod    # 静态方法装饰器   里面不传入任何参数
    def award():
        print(f'获得了12金币, 一件蓝色装备')

# 类方法调用
"""
类方法调用规则, 推荐用类去调用, 尽管实例对象也能调用
用实例对象调用的时候, 也是传到实例对象所对应的类
"""
rush.finish_goal()                                       # 可以调用的原因:  cls 是一个类对象 实参就是前面的rush

p1 = rush("法师", 60, "法杖")                             # 实例化会调用初始化方法
p1.finish_goal("获得装备")                                # 用实例对象调用

# 静态方法调用
"""
静态方法调用规则, 推荐用类去调用, 尽管实例对象也能调用
用实例对象调用的时候, 也是传到实例对象所对应的类
"""
rush.award()

p1 = rush("法师", 60, "法杖")
p1.award()

# 例子
""" 下面两个不同 原因: 初始化方法(__init__)是自动调用的, 好处--不用写重复内容; 普通对象方法(init)不会自动调用 """
def init(self, age):        # self表示当前调用的实例对象
    self.age = age
    print(f'我{self.age}岁开始学习')

def __init__(self, age):
    self.age = age
    print(f'我{self.age}岁开始学习')

# 静态方法和类方法的区别 转至后续继承内容

# 判断
class go_trip:
    def __init__(self, name, adres):
        self.name = name
        self.adres = adres
    
    def trip(self):
        life_object = "相机"
        print(f'{self.name}拿了{life_object},{self.transportation}去了{self.adres}旅游.')

    def transportation1(self):
        self.transportation = "开车"
        print(f'{self.name}今天出行的方式是{self.transportation}')
    
    def transportation2(self):
        print(f'{self.name}今天出行的方式是{self.transportation}')

traveler1 = go_trip("xxx", "西湖")
traveler2 = go_trip("xx", "苏州")

traveler1.transportation = "开飞机"        # 添加了transportation 不会报错
traveler1.trip()                 # 单独不加参数可以调用 会报错, 原因: transportation 没有定义
traveler2.trip()                 # 报错  原因: transportation 是1 独有的, 而2 没有, 所以2 会报错

traveler2.transportation1()      # 当调用1 有了transportation属性 之后,  再调用2 不会报错
traveler2.transportation2()      # 单独调用报错 原因: transportation2 里面没有transportation

traveler1.transportation1()      
traveler2.transportation2()      # 1 调用1 后  2 调用 2报错, 原因: 两个是独立的, 2没有交通工具变量

traveler2.transportation = "步行"    # 动态定义  还是会以最后的赋值结果为准
traveler2.transportation1()          # 执行过程中又重新赋值为: 开车  

traveler2.transportation = "步行"    # 动态定义
traveler1.transportation1()          # 开车      
traveler2.transportation2()          # 步行   和 1 是独立的 ! ! !

class go_trip:
    def __init__(self, name):
        self.name = name

people = go_trip("xxx")
print(go_trip.name())           # 报错, 原因: 类变量无法调用实例对象

people = go_trip("xxx")
go_trip.name = "YYY"
people.name = "zzz"
print(go_trip.name)             # YYY   优先级问题, 实例对象既可以调用实例属性 也可以调用类属性, 当两种同时存在的时候, 类属性调用时只能改变类变量
print(people.name)              # zzz   实例变量优先调用自己, 而不调用类变量   当上面初始化不存在name时, 这一行输出和上面一样
go_trip.adres = "云南" 

# 与属性操作相关的内置函数          (2319 ~~~ 2355)
""" delattr(object, name)  删除 object 的 name 属性(name 参数为字符串）"""
class Person:
    eat = "rice"

    def __init__(self, age):
        self.age = age

p = Person(18)
print(p.eat)
delattr(Person, "eat")          # <=> del Person.eat      这里 eat 必须为字符串
print(p.eat)                    # 已经找不到 eat了

print(p.age)
delattr(Person, "age")          # <=> del Person.age 
print(p.age)

""" getattr(object, name, default(默认值可不写))  返回name对象的属性值, 如果 name 属性不存在，且提供了 default 值，则返回它 """
class Person:
    eat = "rice"

    def __init__(self, age):
        self.age = age

print(getattr(Person, "age", 18))       # 找不到 对象属性值 则返回默认值
print(getattr(p, "age"))                # p 有值 为 18 

""" hasattr(object, name)  判断对象中是不是有这个属性, 有返回True, 无返回False """
print(hasattr(Person, "eat"))           # True
print(hasattr(p, "eat"))                # True
print(hasattr(Person, "name"))          # False

""" setattr(object, name, value)   将对象的name 属性设置成 value这个值, 如果没有就新增 """
setattr(p, "eat", "noodles")            # 和改变量一样
print(p.eat)
setattr(Person, "eat", "noodles")       # 和改变量一样
print(Person.eat)

# 多态性----填坑内容     (2357 ~~~ 2404)
lis = [1, 2, 3, 4, 5]       # 实例变量  
lis.append(6)               # 实例对象调用   没有定制初始化__init__ 所以返回None
print(lis)

lis = list((2, 3), (3, 4), (4, 5))      # 实例化
lis.append(6)                           # lis为实例对象, 所以可以进行调用

dict.items()                # 类方法  可以用实例对象调用       还是实例化, 定义初始化方法

# 封装   在属性或者方法前加两个下划线(__)并声明为私有方法
"""
私有属性或者私有方法只能在类的内部调用, 不能在类外部直接用
可以提供公有方法来访问私有属性, 或者调用私有方法
子类无法继承父类的私有属性和私有方法
总结: 只要类里面加两下划线, 外面就不能调用
"""
class Go_trip:  

    __trsp = "飞机"
    __trsp1 = "火车"

    def __init__(self, name, adres):
        self.name = name
        self.__adres = adres            # 实例变量私有化, 只能在类内部调用

    def __trip(self):                   # 对象方法私有化, 只能在类内部调用
        return f'{self.name}去了北京旅游.'
    
    def ads(self):
        return self.__adres
    
    def trip(self):
        return f'坐{self.__trsp}, {self.__trip()}'

    @classmethod
    def trsp1(clas):
        return  clas.__trsp1

tra1 = Go_trip("白茫茫", "甘肃")    # 实例化, 实例化会调用初始化方法
print(tra1.name)
print(tra1.__adres)          # 报错 对象方法私有化, 只能在类内部调用
tra1.__trip()                # 报错 原因同上

""" 如何调用封装的实例变量: 在类里面加一个中间方法(公有方法) asd 从而得到 """
print(tra1.ads())            # 甘肃 通过类内的中间方法(公有方法)调用, 下面可以调用同理
print(tra1.trip())           # 坐飞机, 白茫茫去了北京旅游. 这里同时调用了方法 和 类变量
print(tra1.trsp1())          # 火车

# 继承          (2406 ~~~ 2516)
"""
单继承:
所有类都默认继承object, 一般不写出来
子类继承父类中非私有的所有属性和方法
作用: 简化代码(子类视角)  功能的扩充(父类视角)
顺序: 先找自己的, 之后找父类, 再之后找父类的父类, 再没有就报错
"""
class Games:

    state = "all_type"

    def area_map(self):
        print("地图") 
    
    def time(self):
        print("多条时间线变化")

    def npc(self):
        print("虚拟程序化人物") 

class Open_world(Games):

    def world(self):
        print("开放世界游戏")

class Story(Open_world):

    def story(self):
        print("开放大世界游戏一般有丰富的主线剧情内容")

class Fps(Games):

    def fire(self):
        print("射击类游戏")

""" 总结: 实例对象不能直接调用父类的类属性, 只能直接调用父类的类方法, 类属性需要通过子类的方法来调用 """

ple = Open_world()
print(Open_world.state)              # 子类调用父类属性
ple.world()                           # 子类优先调用自己
ple.area_map()                        # 子类调用父类的类方法
ple.time()                            # 子类调用父类的类方法

s1 = Story()
s1.story()                           # 自己调用自己方法
s1.world()                           # 调用父类的方法----开放世界游戏
s1.area_map()                        # 调用父类的方法
print(Story.state)                   # 调用父类的父类的方法: 只能通过子类的方法去调用父类里面的类变量

"""
多继承
顺序: 先吃自己的, 没有再去吃父类的, 父类的没有, 吃父类的父类的, 之后照着一条链吃完, 没有再去吃第二个父类之后再链式挨个吃一遍; 
总结: 薅着一个林子往完吃, 吃完这个林子再去下个林子以此类推
"""
class Games:

    state = "all_type"

    def player(self, pl):
        self.pl = pl

    def area_map(self):
        print("地图") 
    
    def time(self):
        print("多条时间线变化")

    def npc(self):
        print("虚拟程序化人物") 

class Open_world(Games):

    def world(self):
        print("开放世界游戏")

class Story(Open_world): 

    def story(self):
        print("游戏一般有丰富的主线剧情内容")

class Fps(Games):

    def fire(self):
        print("射击类游戏")

class Fps_pve(Fps, Story):    # 继承多个父类
    def story_fps(self):
        print("有剧情与NPC斗争的枪战类游戏")

f1 = Fps_pve()                  
f1.story()                    # 调用Story 父类的方法
f1.npc()                      # 这里调用 Fps 类的父类里面的npc, 调用父类的父类 npc 的方法  第一个没有找第二个
f1.time()                     # 先梯度查找, 之后纵向查找

""" 子类无法直接调用父类中私有的类方法和属性, 子类在父类中调用是可以调用的, 与前面私有的内容在类中调用不冲突 """
class Games:
    def __init__(self):
        self.__fire()       # 实例化的时候, 这里可以调用
    
    def __fire(self):
        print("射击类游戏")

    def story_fps(self):
        print("有剧情与NPC斗争的枪战类游戏")

class Story(Games):  
    def story(self):
        print("游戏一般有丰富的主线剧情内容")

s1 = Story()            # 可以调用掉, 因为实例化的时候通过初始化定制, 调用初始化方法      
s1.__fire()             # 报错, 调用不到  不能直接调用, 需要通过中间方法(公有方法)

# 方法重写    (2518 ~~~ 2565)  在继承中, 父类的功能不满足时, 可以在子类中重写父类的方法
class Games:
    def __init__(self, type_game):
        self.type_game = type_game    

    def story_fps(self):
        print(f"{self.type_game}射击类游戏的故事线")

class Story(Games):  
    def story_fps(self):
        print(f"{self.type_game}游戏一般有丰富的主线剧情内容")
        super().story_fps()                     # 相当于 super(Story, self).story_fps
class Good(Story):
    def story_fps(self):
        print("这是个好剧情")
        super().story_fps()                     # 调用当前类的父类story_fps 方法
        super(Good, self).story_fps()           # 调用当Good类的父类story_fps 方法 等同于上面一行
        super(Story, self).story_fps()          # 调用当Story类的父类story_fps 方法

class Nostory(Games):
    def story_fps(self):
        print(f"{self.type_game}主线剧情可以玩")

s1 = Story("PVE射击游戏")       # 实例化, 改变了父类中的一些不满足的内容
s1.story_fps()

s2 = Nostory("PVP射击游戏")     # 改变了父类中的一些不满足的内容
s2.story_fps()

# super  是内置的类, 调用指定类的父类 以上面代码为例 
s1 = Story("PVE射击游戏") 
s1.story_fps()                  # PVE射击游戏游戏一般有丰富的主线剧情内容 + PVE射击游戏射击类游戏的故事线  调用了父类中和子类中的内容

s3 = Good("PVE射击游戏")
s3.story_fps()                  # 优先调用自己 输出: 这是个好剧情
"""
super 单继承: 必须要传是哪个的子类, 否则报错
super().story_fps()               调用父类            PVE射击游戏游戏一般有丰富的主线剧情内容 + PVE射击游戏射击类游戏的故事线    
super(Good, self).story_fps()     调用父类            PVE射击游戏游戏一般有丰富的主线剧情内容 + PVE射击游戏射击类游戏的故事线   
super(Story, self).story_fps()    调用父类的父类       PVE射击游戏射击类游戏的故事线  
"""

""" super 多继承: 横向逐一吃, 优先横向(从左到右)继承, 之后纵向 """

# 类方法和静态方法的区别    (2567 ~~~ 2586)
class A:
    var1 = 123

    @classmethod
    def func1(cls):
        print(cls.var1)

    @staticmethod
    def func2():
        print(A.var1)

class B(A):

    var1 = 312

A.func1()    # 123       
A.func2()    # 123
B.func1()    # 321      上面的cls 变成了 B 类, 他的调用者是谁就是谁
B.func2()    # 123      静态方法没有cls 只能写死, 不管调用者是谁都是输出固定类的类变量或者类方法

# 继承中初始化方法      (2588 ~~~ 2621)
class A:

    def E(self):
        print('E方法被调用')

    def __init__(self, name):
        self.name = name
        self.Q()

    def Q(self):
        print(self.name, 'Q方法被调用')

class B(A):
    pass

class C(A):
    def __init__(self, name):
        self.names = name

class D(A):
    def __init__(self, name):
        super(D, self).__init__('李四')
        self.name = name

b = B("三三")
b.E()
b.Q()           # Q 被调用了2 次, 在实例化的时候调用了一次    如果self.name = name 和 self.Q 交换 会报错, 没有定义

c = C("四四")
c.Q()           # 报错, 因为有限使用自己的定制化(初始化) 所以没有引用到A 里面的

d = D("王舞")   # 李四 Q方法被调用   原因: 调用A 里面初始化方法, 所以会先变成李四, 之后根据父类A里面的初始化内容运行  A 类的self 是D 的self
d.Q()           # 王舞 Q方法被调用   原因: 先执行了super 之后又执行了下一行, 所以变量又改了回来

# 与继承相关的内置函数   (22623 ~~~ 2653)
""" isinstance(object, classinfo)     object: 实例对象; classinfo: 类名
作用: 如果object 是 classinfo 的实例对象或者是子类 返回True  
判断实例对象或者子类在不在类之内 """
class A:
    pass
class B(A):
    pass
class C(A):
    pass

a = A()
b = B()
c = C()
print(isinstance(a, A))         # True   判断 a 是不是 A 的实例对象
print(type(a) == A)             # True  

print(isinstance(b, A))         # True   因为考虑继承关系 所以为True
print(type(b) == A)             # False  type 不考虑继承

print(isinstance(c, A))         # True   因为考虑继承关系 所以为True
print(type(c) == A)             # False  type 不考虑继

print(isinstance(c,(A, B)))     # True   C 是 A 子类的实例
print(isinstance(c, A) or isinstance(c, B))    # 上面相当于这样

""" issubcalss(class, classinfo) 以上面代码为例 """
print(issubclass(A, B))         # False 判断A 是不是 B 的子类
print(issubclass(B, A))         # True 判断B 是不是 A 的子类
print(issubclass(C, (A, B)))    # True 相当于print(issubclass(c, A) or issubclass(c, B))
print(issubclass(A, A))         # True 类会被其自身判为子类

# 多态性 (2655 ~~~ 2675)  不同的内容方法可以使用相同的方法名
class Apple:
    def change(self):
        return '啊~ 我变成了苹果汁!'
class Banana:
    def change(self):
        return '啊~ 我变成了香蕉汁!'
class Mango:
    def change(self):
        return '啊~ 我变成了芒果汁!'
class Juicer:
    def work(self, fruit):
        print(fruit.change())

a = Apple()
b = Banana()
m = Mango()
j = Juicer()
j.work(a)               # 三个不同类的方法用相同名字定义
j.work(b)               # 可以通过改变方法的调用的对象, 调用不同内容
j.work(m)               # 必须有一个中间的类去调用它

# 学生和老师的一天(魔改其他版本)   (2677 ~~~ 2745)
class Trip:

    count = 0

    def __init__(self, name ,suggestion, trsp):
        self.name = name
        self.suggestion = suggestion
        self.trsp = trsp
        self.area()
        Trip.count += 1                                             # 公共部分

    def area(self):
        print(f"{self.name}, 每人提出一个{self.suggestion}, 怎么去那里: {self.trsp}")
        
    def sleep(self):
        print(f'{self.name}睡觉')

class mountain(Trip):
    count = 0
    def __init__(self, name, suggestion, trsp, house):               # 可以把子类独有放在子类中
        super().__init__(name, suggestion, trsp)
        self.house = house
        mountain.name = self.name 
        mountain.count += 1

    def tent(self):
        print(f'{self.name}拿出帐篷')
        print(f'{self.name}搭帐篷')                                 # 山中部分
    
    def driver(self):
        print(f'{self.name}开车')
    
    def cooking(self):
        print(f'{self.name}洗菜')
        print(f'{self.name}做饭')
    
    @classmethod                                                        # 改成静态 @staticmethod
    def counter(cls):                                                   # def counter():
        print(f"当前{cls.name} 做了{cls.count}事")                       # print(f"当前{mountain.name} 做了{mountain.count}事") 

class ocean(Trip):
    count = 0
    def __init__(self, name ,suggestion, trsp):
        super().__init__(name, suggestion, trsp)
        ocean.count += 1

    def takes(self):
        print(f'{self.name}拿出遮阳伞')
        print(f'{self.name}拿出毯子')
    
    def driver(self):                                                   # 海滩部分
        print(f'{self.name}坐火车')
    
    def eat(self):
        print(f'{self.name}吃冷饮')
        print(f'{self.name}喝冰椰子汁')
    
    @classmethod                                                        
    def counter(cls):                                                 
        print(f"当前做了{cls.count}事")                       

tra1 = Trip("伍伍", "需要带帐篷和手电", "开车去")
tra2 = mountain("陆陆", "拿上驱虫药和绷带", "坐火车去", "林中小屋")
tra1 = ocean("柒柒", '需要带遮阳伞和毯子', "坐火车去")
tra2 = ocean("捌捌", "拿冲浪板, 手机", "坐大巴")
tra1.sleep()
mountain.counter()
ocean.counter()

# 魔术方法  (2747 ~~~ 2825)    __i__ 特殊方法, 特殊在特殊情况下可以自动调用
""" __init__初始化方法"""
class Ex:
    def  __init__(self, name):
        print(f'{name}由__init调用')
Ex('a')         # 实例化 调用初始化方法

"""__call__ 当实例对象像函数那样被调用时, 会调用该方法 """
class Ex:
    def  __call__(self, name):
        print(f'{name}由__init调用')
e = Ex()     
e("a")
# 相当于
Ex()("a")       # 调用实例对象

# 重视问题
a = [1, 2, 3]       # 实例对象
tup = tuple(a)
print(a(tup))       # 用a 这个实例对象像函数那样去调用, 会去调用a类, 即(list类) 的__call__的特殊方法
print("__call__"in dir(list))       # False 说明在list 里面么有 __call__  所以无法这样调用

""" __getitem__(self, key) 执行self[key]时, 会自动调用"""
class Ex:
    def  __getitem__(self, key):
        print(f'__getitem__被调用, key: {key}')                            # key 提示器
        print(["a", "b", "c", "D", "e"][key])
        # print({0: "零", 1: "壹", 2: "二", 3: "三", 4: "四"}[key])         # 这个不能切
a = Ex()
a[1: 3]             # 索引可以是值可以是字典的键 还可以切片

""" __repr__(self) / __str__(self) 实例对象转字符串的时候, 会调用该方法, 需求必须返回字符串类型"""
class Ex:
    def  __repr__(self):
        return '__repr__被调用'             
e = Ex()
print(str(e))
print(f'{e}')
print(e)                    # 三个东西一个意思, 必须转成字符串

""" 
__add__(self, other)        进行加法操作的时候调用, 要求实例对象在两对象想加的左边
__radd__(self, other)       进行加法操作的时候调用, 要求实例对象在两对象想加的右边, 而且左边不为实例对象
__sub__(self, other)        进行减法操作的时候调用, 要求实例对象在两对象想减的左边
__rsub__(self, other)       进行减法操作的时候调用, 要求实例对象在两对象想减的右边, 而且左边不为实例对象
__mul__(self, other)        进行乘法操作的时候调用, 要求实例对象在两对象想乘的左边
__rmul__(self, other)       进行乘法操作的时候调用, 要求实例对象在两对象想乘的右边, 而且左边不为实例对象
__truediv__(self, other)    进行除法操作的时候调用, 要求实例对象在两对象想除的左边
__rtruediv__(self, other)   进行除法操作的时候调用, 要求实例对象在两对象想除的右边, 而且左边不为实例对象
"""
class Number:
    def __init__(self, num):
        self.num = num
    def __add__(self, other):
        return self.num + other
n = Number(6)
print(n + 7)                # 实例对象在左边  加减乘除同理

class Number:
    def __init__(self, num):
        self.num = num
    def __radd__(self, other):
        return other + self.num
n = Number(6)
print(7 + n)                # 要求左边不为实例对象  加减乘除同理

class Number:                       # 同时存在
    def __init__(self, num):
        self.num = num

    def __add__(self, other):
        return self.num + other

    def __radd__(self, other):
        return other + self.num
        
n1 = Number(6)               # 先跳至 return self.num + other 在算完结果后 跳至 return other + self.num 之后得到的左边的数是一个int 之后再对 n2 进行实例化
n2 = Number(7)
print(n1 + n2)  


# 实现分数的运算  (2827 ~~~ 2910)

"""前提条件: 最大公约数"""
def gcd(nt, dn):                        # 找出两个数的最大公约数
    for i in range(min(abs(nt), abs(dn)), 0, -1):
        if not nt % i and not dn % i:
            return -i if nt < 0 and dn < 0 else i
    return dn

def get_fraction(obj):                  # 如果有个数为常数, 把他转化为分数形式
    if isinstance(obj, Fraction):
        return obj                      # 如果类型是分数, 返回这个数
    elif isinstance(obj, int):
        return Fraction(obj, 1)         # 如果不是, 则返回分数的这个形式  i / 1 类型
    elif isinstance(obj, float):
        dn = 10**(len(str(obj).split(".")[-1]))     # 把小数变成分数
        return Fraction(int(obj*dn), dn)          # 小数变成分数
    raise TypeError("类型错误")
    
class Fraction:
    def __init__(self, nt, dn):
        gcd_num = gcd(nt, dn)
        self.nt = nt // gcd_num
        self.dn = dn // gcd_num
    
    def __str__(self):                   # 实现分数的输出
        if not self.nt:
            return "0" 
        elif self.dn == 1:
            return str(self.nt)
        elif self.dn == (-1):
            return str(-self.nt)
        return f'{self.nt}/{self.dn}'    # 分子nt / 分母

    """ 两个分数相加, 第一个分子乘以第二个的分母 + 第二个的分子乘以第一个的分母, 第一个的分母乘以第二个的分母
    会自动调用__add__魔术方法之后会调用初始化(__init__)方法"""
    def __add__(self, other):            # 实现分数的加法
        other = get_fraction(other)
        return Fraction(self.nt * other.dn + other.nt * self.dn, self.dn * other.dn)        # 这里返回实例对象的时候 f1 + f2 返回就是一个实例对象  

    def __sub__(self, other):
        other = get_fraction(other)
        return Fraction(self.nt * other.dn - other.nt * self.dn, self.dn * other.dn)        # 与加法同理

    """ 两个分数相乘, 分子乘分子, 分母乘分母 """
    def __mul__(self, other):
        other = get_fraction(other)
        return Fraction(self.nt * other.nt, self.dn * other.dn)

    """ 两个分数相除, 第一个分子乘第二个分母, 第二个分子乘第一个分母"""
    def __truediv__(self, other):
        other = get_fraction(other)
        return Fraction(self.nt * other.dn, self.dn * other.nt)

    """ 对左边为固定数的内容进行加减乘除 """
    def __radd__(self, other):
        other = get_fraction(other)
        return Fraction(other.nt * self.dn + other.dn * self.nt, other.dn * self.dn)

    def __rsub__(self, other):
        other = get_fraction(other)
        return Fraction(other.nt * self.dn - other.dn * self.nt, other.dn * self.dn)
    
    def __rmul__(self, other):
        other = get_fraction(other)
        return Fraction(other.nt * self.nt, other.dn * self.dn)

    def __rtruediv__(self, other):
        other = get_fraction(other)
        return Fraction(other.nt * self.dn, other.dn * self.nt)

f1 = Fraction(4, 1)
f2 = Fraction(1, 2)
print(f1 + f2)   
print(f1 - f2)  
print(f1 * f2)
print(f1 / f2)
print(f1 + 1.5)
print(4 + f1)
print(4 + f2)   
print(4 - f2)  
print(4 * f2)
print(4 / f2)

# 闭包  (2913 ~~~ 2959)
""" 
闭包的构成条件:
1. 是个嵌套函数
2. 内部使用了外部的变量或者参数
3. 外部函数的返回值是内部函数的引用
"""
def outer():                # 条件1: 嵌套函数
    a = 3
    def inner():       
        b = a + 4           # 条件2: 内部函数值引用外部变量或者参数
        return b
    return inner            # 条件3: 外部函数的返回值是内部函数的引用

print(outer())              # 函数的地址
print(outer()())            # 调用inner函数

""" 一般来说函数结束时, 里面内变量, 参数会释放掉;
闭包不会释放, 他会在外部函数结束的时候, 把内部用到的外部函数变量, 参数保存到内部函数的__closure__属性中, 提供内部函数使用"""
def outer(d):                # 条件1: 嵌套函数
    a = 3
    c = 4
    def inner():       
        b = a + c + d        # 条件2: 内部函数值引用外部变量或者参数
        return b
    return inner             # 条件3: 外部函数的返回值是内部函数的引用

inner_f = outer(5)
print(inner_f.__closure__[0])      # 这里是两个地址(a, c), 因为引用了两个外部变量, 所以会保存到内部函数的__closure__属性中
""" 总结: 内部函数引用多少个外部变量就有多少个地址, """
print(inner_f.__closure__[0])                       # 查看他存的地址(索引)
print(inner_f.__closure__[0].cell_contents)         # 查看他存的内容(索引)
print(inner_f())                                    # 之后可以拿来直接用

# 栗子:
def outer():
    funcs = []
    for k in range(3):          # 循环 3 此 闭包函数: 他会在外部函数结束的时候, 把内部用到的外部函数变量, 参数保存到内部函数的__closure__属性中, 提供内部函数使用
        def inner():
            return k * k
        funcs.append(inner)
    return funcs

f1, f2, f3 = outer()            # 解包  2 2 2 保存到funcs 中
print(f1())                     # 4     外部函数结束时 结果为4 把4保存到__closure__ 中
print(f2())                     # 4     外部函数结束时 结果为4 把4保存到__closure__ 中
print(f3())                     # 4     外部函数结束时 结果为4 把4保存到__closure__ 中

# 装饰器        (2961 ~~~ 2979)
"""
作用: 使代码结构更加清晰, 内容更加优雅
将待定功能代码封装成装饰器, 提高代码复用率, 增强代码可读性
在原函数上增加些功能
总结: 只有头上加了装饰器的方法或者函数才能进行出现装饰器的作用
"""
# 栗子: 
class A:

    @classmethod        # 变成类方法 使类可以调用
    def func(cls):
        print(123)

    def func2(cls):
        print(456)

A.func()        # 可以调用        
A.func2()       # 不能调用

# 函数装饰器(2981 ~~~ 3119)
# 不带参数的函数装饰器
import time

def timer(f):
    def sometime(s_timer):
        start = time.time()   # 返回当前的时间
        f(s_timer)
        end = time.time()
        print(f'函数执行花费{end - start}秒')
    return sometime

""" 
当被装饰好的函数(func) 定义好时, 下一步会把它作为参数传给装饰器函数并调用,  即timer(func) 想当于 下面func1 = timer(func) 
然后返回 timer 函数, 再定义sometime, 再返回 sometime (相当于func1 = timer(func) 调用sometime)
最后执行func(2) 其实就是执行 sometime(2)    
运行过程: 先把到大框架运行完, 之后运行传入参数
"""
@timer                       # 装饰器, 是一个函数, 定义装饰器函数  
def func(s_timer):           # 函数名作为参数传给装饰器
    time.sleep(s_timer)
    print("执行func花费的时间")

func(2)

# func1 = timer(func)     还可修改
# func1(2)

# start = time.time()   # 返回当前的时间
# func()
# end = time.time()
# print(f'函数执行花费{end - start}秒')


# 带参数的函数装饰器
# 第一种, 可以执行 不是很完美
def timer(f):
    def wrapper(s_time, name):
        start = time.time()
        f(s_time)
        end  = time.time()
        print(f'{name}, 衣服洗好了, 洗衣服花费时间{end - start}秒, 快来拿啊!~')
    return wrapper

@ timer
def washer(s_time):
    time.sleep(s_time)
    print("didididid")

washer(3, "三三")

# 第二种
def interaction(name):              # 交互   传入过实参
    def timer(f):
        def wrapper(s_time):
            start = time.time()
            f(s_time)
            end  = time.time()
            print(f'{name}, 衣服洗好了, 洗衣服花费时间{end - start}秒, 快来拿啊!~')
        return wrapper
    return timer

""" 当被装饰的函数定义好之后, 会把washer(定义好的函数)作为参数传到装饰器里, 执行后并返回 的函数 同时调用该函数, 即timer 👈把washer 放进去
之后定义wrapper 并返回
"""
@ interaction("三三")               # 返回timer 定义一个timer
def washer(s_time):                 # 当被装饰的函数定义好之后, 会把washer作为参数传到装饰器里
    time.sleep(s_time)
    print("didididid")

""" 问题, 既然washer 地址是wrapper 那么为什么washer 函数里面的内容还会被执行:   在上面wrapper 里有 f(s_time) 相当于调用了washer """
washer(3)    # 即 wrapper(3)

# 类装饰器
import time

# 在类中初始化传 f  (不够完美)
class Timer:                            # 定义类  之后定义函数

    def __init__(self, f, name):              # 定义类里的函数按顺序, 之后到 class Timer    这里的 f 是 washer
        self.func = f
        self.name = name

    def __call__(self, s_time):
        start = time.time()
        self.func(s_time)
        end  = time.time()
        print(f'{name}衣服洗好了, 洗衣服花费时间{end - start}秒, 快来拿啊!~')

@Timer    # 进行实例化; 定义完类后, 返回calss Timer 
def washer(s_time):    # 返回一个 __new__    obj = Timer(washer)       被装饰的函数定义好后把 它的名字作为参数传给装饰器 并调用装饰器            
    time.sleep(s_time)
    print("didididid")

washer(2, "三三")   # obj()      然后 把实例对象返回给 washer   相当于返回了一个实例对象 之后计算函数内容

# 带参数的类装饰器(优化版本)
class Timer:                            # 定义类  之后定义函数

    def __init__(self, name):              # 定义类里的函数按顺序, 之后到 class Timer    
        self.name = name

    def __call__(self, f):      # f 为 washer
        def res(s_time):
            start = time.time()
            f(s_time)
            end  = time.time()
            print(f'{self.name}衣服洗好了, 洗衣服花费时间{end - start}秒, 快来拿啊!~')
        return res              # 返回 washer

@Timer("三三")    # 实例化 返回一个实例对象  obj = Timer("三三")
def washer(s_time):    # 调用实例对象 被装饰的函数定义好后把 它的名字作为参数传给装饰器 并调用装饰器   res =  obj(washer)    转至 3083 行       调用__call__返回res
    time.sleep(s_time)
    print("didididid")

washer(2)   # res(2)      相当于__call__ 的返回值, 调用res 内容  

# 嵌套类
class Timer:                            # 定义类  之后定义函数

    def __init__(self, name):              # 定义类里的函数按顺序, 之后到 class Timer    
        self.name = name                   # 这里的self 是obj

    def __call__(self, f):      # f 为 washer
        name = self.name
        class res:
            def __init__(self, s_time):
                start = time.time()
                f(s_time)
                end  = time.time()
                print(f'{name}衣服洗好了, 洗衣服花费时间{end - start}秒, 快来拿啊!~')      # 这里没有name  这里的对象是res 
        return res

@Timer("三三")    # 实例化 返回一个实例对象  obj = Timer("三三")
def washer(s_time):    # 调用实例对象 被装饰的函数定义好后把 它的名字作为参数传给装饰器 并调用装饰器   res =  obj(washer)    转至 3083 行       调用__call__返回res
    time.sleep(s_time)
    print("didididid")

washer(2)

# 多个装饰器    (3121 ~~~ 3153)
""" 有多个装饰器, 后进的先出, 先进的后出 例如下面的, 会先执行timer 后执行deco 和函数的定义位置无关, 和装饰器的位置有关"""
import time 
def deco(func):             # func 是: wrapper2

    def wrapper1(*args):    # args: (3, 4, 5)
        res = func(*args)   # wrapper2(3, 4 ,5)    *args: 解包
        return res          # res = 12  👈 由 wrapper2 得到
    return wrapper1

def timer(func):            # func 是: add

    def wrapper2(*args):    # arg: (3, 4, 5)
        start = time.time()
        res = func(*args)   # add(3, 4, 5)   *arg: 解包  res = 12 
        end  = time.time()
        print(f'函数耗时: {end - start}')
        return res          # 返回 12 
    return wrapper2

"""
wrapper2 = timer(add)   返回wrapper2
deco(wrapper2)  返回(=>) wrapper1
add = wrapper1
add(3, 4, 5) => wrapper1(3, 4, 5)
"""
@deco     # 后调用的
@timer    # 先调用的
def adds(*args):
    time.sleep(2)
    return sum(args)

print(adds(3, 4, 5))

# 内置装饰器    (3155 ~~~ 3179)  @classmethod  @staticmethod  @property(属性)
""" @property 只能对这个属性读取, 不能修改"""
class Student:

    def __init__(self, name, age):
        self.name = name
        self.age = age 

    @property
    def adult_age_p(self):
        return 21
    
    def adult_age(self):
        return 21

    @property
    def adult_flage(self):
        return self.age >= self.adult_age_p

stu = Student("亖亖", 21)    # 实例化
print(stu.adult_age())      # 没有加 @property, 必须使用正常的调用方法的形式, 在后面加 ()
print(stu.adult_age_p)      # 加 @property. 用调用属性的形式调用方法, 后面不需要加()
print(stu.adult_flage)      # True
stu.age = 17                # 可改
stu.adult_age_p = 17        # 报错 只读属性 只能看不能改


# 错误分类    (3182 ~~~ )
# 查看内置异常    (3183 ~~~ 3296)
import builtins
print(dir(builtins)) 

# 处理异常
# try:    except:     except 可以指定异常类型, 没有指定则处理所有异常
try:                 # 当try 内容发生异常时候 执行 except 内容
    4 / 0
except:
    print(1234)

# 栗子 🌰
def div(a, b):  
    try:
        c = a / b
        print(f'{a} / {b} = {c}')
    
    except:                              # 下面两个运行时 这个需要注释掉  否则 error: default 'except:' must be last
        print("try中发生异常")

    # except ZeroDivisionError:
    #     print("try中发生了除零错")

    # except TypeError:
    #     print("try中出现了类型异常")

    # except(ZeroDivisionError, TypeError):       # 运行这个前面都需要注释掉
    #     print("try中发生了除零错误或者异常类型")

div(2, 1)       # 2 / 1 = 2.0
div(2, 0)       # try中发生异常 
div("2", 2)     # try中发生异常

# try:    except:  嵌套    没有指定错误的时候, 会直接运行接受所有类型错误的 excpet
def div(a, b):  
    try:
        try:
            c = a / b
            print(f'{a} / {b} = {c}')

        except ZeroDivisionError:
            print("try中发生了除零错")

    except:                                
        print("try中发生异常")

div(2, 1)       # 2 / 1 = 2.0
div(2, 0)       # try中发生了除零错
div("2", 2)     # try中发生异常

# try:  except:  else:   else 就是在try 里面没有发生异常的时候执行
def div(a, b):  
    try:
    
        c = a / b
        print(f'{a} / {b} = {c}')

    except ZeroDivisionError:
        print("try中发生了除零错")

    except:                                
        print("try中发生异常")

    else:
        print("try中没有异常")

div(2, 1)       # 2 / 1 = 2.0
div(2, 0)       # try中发生了除零错
div("2", 2) 

# try:  except:  as:    as 作为后面异常实例对象的名称
def div(a, b):  
    try:
    
        c = a / b
        print(f'{a} / {b} = {c}')

    except ZeroDivisionError as e:    # 相当于 e = ZeroDivisionError('division by zero')
        print(e, ZeroDivisionError)    # division by zero <class 'ZeroDivisionError'>
        print(e)    # division by zero
        print(ZeroDivisionError('division by zero'))    # division by zero

div(2, 0)    

# try finally       不论 try 里的内容运行是否异常, 都会去执行
def div(a, b):  
    try:
    
        c = a / b
        print(f'{a} / {b} = {c}')

    except:                                
        print("try中发生异常")

    else:
        print("try中没有异常")
    
    finally:
        print("执行了finally子句")

div(2, 1)       # try中没有异常   不报错也会执行finally 子句    try 👉 else 👉 finally
div(2, 0)       # try中发生异常   在报错前执行finally 子句      try 👉 except 👉 finally

# finally 有返回值, 结果是finally 的return
def return_num():
    try:
        return 1

    finally:
        return 2

print(return_num())       # 2   不论try 里面发生了什么 是结束还是 break finally都会执行 所以返回 2 

# 抛出异常  (3296 ~~~ 3306)
# raise 会主动抛出异常, 后面可以是实例对象, 也可以是类 会把报错的内容改成想要的内容, 还可以没有内容
def div(a, b):
    if b == 0:
        raise ZeroDivisionError("除数为0")
    
    c = a / b
    print(f'{a} / {b} = {c}')

div(2, 1)       # 不会触发 raise
div(2, 0)       # 除数为0 执行raise 的结果   ZeroDivisionError: 除数为0 

# 自定义异常    (3308 ~~~ 3352)
# 继承Exception 类
class MyError(Exception):
    def __init__(self, message):
        self.message = message
    
    def __str__(self):
        return str(self.message)

print(MyError("发生了一个异常"))    # 发生了一个异常  print(ZeroDivisionError("除数为0"))  一回事

# assert 断言 用于判断一个表达式, 表达式为False 的时候触发 AssertError 异常
num = int(input("请输入一个整数:"))
assert num != 1, "用户不能输入1"    # 断言为1  输入的不是1 (成功状态)会继续执行后面内容  会执行后面的东西, 如果是1 就会报断言错误 
print("断言条件为True, 用户没有输入1 ")
"""相当于下面"""
num = int(input("请输入一个整数:"))
if num != 1:
    print("断言条件为True, 用户没有输入1 ")
else:
    raise AssertionError("用户不能输出1")
"""相当于下面"""
num = int(input("请输入一个整数:"))
if not num != 1:
    raise AssertionError("用户不能输出1")
print("断言条件为True, 用户没有输入1 ")

# traceback.print_exc() 和 traceback.format_exc()
""" traceback回溯
两个区别
traceback.print_exc()  无返回值, 直接输出
traceback.format_exc() 返回一个字符串
"""
import traceback
def div(a, b):
    try:
        c = a / b
        print(f'{a} / {b} = {c}')
    except:
        traceback.print_exc()       # 报错逻辑 基于这个实现
        print(res := traceback.format_exc())    # 返回一个字符串, 上面的报错形式
        print(type(res))    # str

div("2", 2)

# 可迭代带对象, 迭代器, 生成器     (3354 ~~~ 3453)
""" 判断iterable iterator generator """

""" 
iterable(可迭代对象)
str list tuple dict set range zip map filter enumerate reversed
"""
from typing import Iterable

string = "abc"
print(isinstance(string, str))      # 是iterable, 也是str类型 原因 : 继承, str 是 iterable 的子类
print(issubclass(str, Iterable))    # str 是 iterable  的子类

""" 文件对象是迭代器 """
with open("文件储存位置", mode = "r") as file:          # 通过 open 的打开这个文件, 以mode 的形式
    print(isinstance(file, Iterable))    # 是可迭代对象
    print(isinstance(file, Iterator))    # 也是迭代器

"""
iterator(迭代器)
zip map filter enumerate reversed
"""
from typing import Iterator
print(issubclass(zip, Iterator))          # True   zip 是 迭代器的子类
print(issubclass(map, Iterator))          # True   map 是 迭代器的子类
print(issubclass(filter, Iterator))       # True   filter 是 迭代器的子类
print(issubclass(enumerate, Iterator))    # True   enumerate 是 迭代器的子类
print(issubclass(reversed, Iterator))     # True   reversed 是 迭代器的子类

"""
生成器(Generator)

"""
from typing import Generator
print(issubclass(str, Generator))

""" 
总结 :
迭代器一定是可的迭代对象, 生成器一定是迭代器
"""

# 可迭代对象
""" 
只要满足下面两个条件之一就是可迭代对象:
支持迭代协议: __iter__() 方法
支持序列协议: __getitem__()方法, 且数字参数从0 开始

iterable(可迭代对象)
str list tuple dict set range zip map filter enumerate reversed

总结: 实现 __iter__()  或者  __getitem__() 之一的就是可迭代对象
""" 

print("__iter__" in dir(str) or "__getitem__" in dir(str))  # True
print("__iter__" in dir(list) or "__getitem__" in dir(list))    # True

# 自己定义可迭代对象
from typing import Iterable 

class A:
    def __iter__(self):
        pass

a = A()
print(isinstance(a, Iterable))    # True 说明设定了个实例对象 a 是可迭代对象
for i in a:     # 单独报错, 没有实现内容
    print(i)

class B:
    def __getitem__(self):
        pass
b = B()
print(isinstance(b, Iterable))   # False 使用 isinstance 无法检测 __getitem__ 是否可以迭代
for i in b:     # 检测方法 , for 变量 in iterable:
    print(i)

# 迭代器(为什么可以迭代)
"""
支持迭代器协议的(区分迭代协议 满足其中一个)  同时满足两个条件 
实现 __iter__()方法  需要和下面同时满足
实现 __next__()方法  需要和上面同时满足

迭代器里面必须包含迭代协议 所以迭代器一定是迭代对象
"""
from typing import Iterable, Iterator
print("__iter__" in dir(list) and "__next__" in dir(list))            # False
print("__iter__" in dir(zip) and "__next__" in dir(zip))              # True
print("__iter__" in dir(reversed) and "__next__" in dir(reversed))    # True 
print("__iter__" in dir(sorted) and "__next__" in dir(sorted))        # False

# 自己创建迭代器  (3444 ~~~ 3570)
class A:
    def __iter__(self):
        pass

    def __next__(self):
        pass

a = A()
print(isinstance(a, Iterator))      # True  a 是迭代器 也是可迭代对象

# 迭代的逻辑
class ListIterable:

    def __init__(self, obj):
        self.obj = obj              # 这个obj是下面类的那个self,  这里的self 是 ListIterable 的 self 和下面不一样
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):             # 魔术方法, 对象方法
        if self.count < len(self.obj.iterable):    
            item = self.obj.iterable[self.count]    # 因为下面有这个参数, 所以在传参后这里可以调用下面的iterable
            self.count += 1
            return item
        raise StopIteration                         # 如何结束, 可以使用 exit()  整个退出python文件 这里抛出错误也可终止

class Mylist:

    def __init__(self, iterable):
        *self.iterable, = iterable
    
    def __str__(self):  
        return f"输出成列表: {self.iterable}"       # 这个位置输出成列表

    def __iter__(self):
        return ListIterable(self)    # 要通过实例对象 自动去调用  ListIterable 的 __next__
    
""" 
1. 当for循环开始遍历的时候, 会自动调用 __iter__ 这个魔术方法
2. __iter__这个魔术方法必须返回迭代器对象, 因为每次循环的时候会用该迭代器对象自动调用__next__() 是 实例对象去调用__next__
3. 每次调用 __next__ 的返回值赋值给变量 i """
mylis = Mylist((1, 2, 3))
print(mylis)

# 手动实现 for 循环
lis_iter = mylis.__iter__()
i = lis_iter.__next__()     # 1 第一次调用
i = lis_iter.__next__()     # 2 第二次调用
i = lis_iter.__next__()     # 3 第三次调用
print(i)
""" 简化"""
while True:
    try:
        i = lis_iter.__next__()
        print(i)
    except StopIteration:
        break

# 对迭代器进行迭代
""" 
① 可迭代对象里的__iter__()方法返回一个迭代器，通过迭代器里的__next__()方法实现迭代
② 迭代器里的__iter__()方法返回它本身（因为迭代器协议中包含了迭代协议，所以迭代器也一定是可迭代对象，可迭代对象的__iter__()方法要返回一个迭代器，所以只需要返回本身即可）
③ 迭代器里的__next__()方法返回可迭代对象的下一项，如果没有下一项可返回，则抛出 StopIteration 异常
"""
class ListIterable:

    def __init__(self, obj):        # self 是 ListIterable 的 self 不是 Mylist 的 self 这里的self 是 Mylist((1, 2, 3, ))  obj 则是 Mylist 的 self
        self.obj = obj              # 这个obj是下面类的那个self,  这里的self 是 ListIterable 的 self 和下面不一样
        self.count = 0

    def __iter__(self):             # 迭代协议
        return self                 # 这里的self 是 ListIterable 的 self  返回了他自己的self 也就是  Mylist((1, 2, 3, ))

    def __next__(self):             # 这里的__next__ 由上面的 __iter__ 的 self 调用, 魔术方法, 对象方法  循环的时候会调用 __next__
        if self.count < len(self.obj.iterable):    
            item = self.obj.iterable[self.count]    # 因为下面有这个参数, 所以在传参后这里可以调用下面的iterable
            self.count += 1
            return item
        raise StopIteration                         # # 如何结束, 可以使用 exit()  整个退出python文件 这里抛出错误也可终止

class Mylist:

    def __init__(self, iterable):
        *self.iterable, = iterable            # 解包 组成列表来赋值
    
    def __str__(self):
        return f"输出成结果: {self.iterable}"

    def __iter__(self):
        return ListIterable(self)             # 这里把Mylist 的实例的对象传给了 ListIterable

for i in ListIterable(Mylist((1, 2, 3, ))):   # 先运行里面的内容: 先传给 Mylist 的 __init__  之后传给 ListIterable 的 obj 再之后 retrun self  接下来连续判断条件 输出想要的结果
    print(i)

# 如何迭代
map_itera = map(abs, [1, -2, 3, 4])
for i in map_itera:
    print(i)

print(list(map_itera))      # 都需要执行 __iter__ 之后进行 __next__ 再把元素提出来构成列表
""" 总结: 函数和方法里面传可迭代对象, 对可迭代对象进行处理的时候, 是基于迭代的逻辑 __iter__ 返回一个迭代器, 之后调用 __next__  依次把元素提出来, 最后改成想要的形式 """
list(map_itera)
""" 把需要迭代的元素构成一个相符的容器, 先会对可迭代对象进行迭代, 把元素一个一个拿出来, 之后构成想要的形式"""
print(sum(map_itera))       # 0 原因: 在上面list(map_itera) 运行之后 map_itera.count = 4  所以 map_itera 为空的, 因此为0
 # 两个连续执行第二个未执行 
for i in map_itera:         # 在执行结束后, map_itera.count = 4 
    print(i)
for i in map_itera:         # 之后执行这个也是从4 开始 所以直接 raise 错误 循环截止
    print(i)    

for i in map(abs, [1, -2, 3, 4]):     # 这个样子会执行两次,  原因: 调用两次是实例化两次, 两次的地址不同, 会返回两个不同的迭代器    
    print(i)

for i in map(abs, [1, -2, 3, 4]):     # 对迭代器迭代的时候会返回self 
    print(i)    

""" 当迭代对象用的是一个迭代器的时他会在第二个开始的时候终止, 原因: 运行的次数会接第一个迭代后增加
当同一个对象进行不同迭代的时候, 产生两个迭代器, 所以两个都会运行 
"""
a = [1, 2, 3]       # 这里为什么会执行两次
for i in a:         # 原因: 两次对可迭代对象 a 进行遍历的时候, 他会调用 两次魔术方法, 相当于返回两次实例化, 所以迭代器不是同一个 
    print(i)

for i in a:         # 上面map 返回迭代器, 所以返回同一个迭代器, 因此上面map 只输出一次
    print(i)

# 自我尝试 (3572 ~ 3615)
import random

class List_ia:

    def __init__(self, lis_name):
        self.lis_name = lis_name
        self.count = 0

    def __call__(self):     
        pass

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.count < len(self.lis_name.typer):
            item = self.lis_name.typer[self.count]
            self.count += 1
            self.string = self.lis_name.__str__
            return f'{self.string()} 和随机抽取的 {item}'
        raise StopIteration         # 迭代器里面无法捕获错误, 需要手动抛出错误

class Change_list:

    def __init__(self, typer):
        *self.typer, = typer        # 解包的时候需要加 " , "

    def __str__(self):
        return f'输出成某种type: {self.typer}'

    def __iter__(self):
        return List_ia(self)        # self 不能忘了 需要把这里的实例变量传给 List_ia → 转为 lis_name

def typ():
    num = zip(random.sample(["name", "解释器", "解释器", "18"], 3), set(random.sample(["vscodde", "使用时间", "使用时间", "代码"], 4)))
    return num
for i in List_ia(Change_list(typ())):
    print(i)

ex1 = {"name": "vscodde", "作用": "解释器", "使用时间": 18}
for i in List_ia(Change_list(ex1)):
    print(i)

# 在前面迭代一半的情况下, 后面迭代的内容会跟着同一个迭代器继续下去
map_itera = map(abs, [1, -2, 3, 4, 5])
print(map_itera.__next__())    # 1
print(map_itera.__next__())    # 2     
for i in map_itera:     # 3, 4, 5
    print(i)

class Mylist:

    def __init__(self, iterable):
        *self.iterable, = iterable

    def __str__(self):
        return f'{self.iterable}'

    def __getitem__(self, index):
        return self.iterable[index]     # 这里灰色等等原因是 上面为解包

mylis = Mylist((1, 2, 3, 4, 5, 6))
for i in mylis:         # 先会找 __iter__ 之后才会招 __getitem__
    print(i)

print(tuple(mylis))
print(set(mylis))
print(sum(Mylist((1, 2, 3 ,4))))

# 生成器
""" 生成器类似于普通函数写法, 
不同点:  生成器用 yield 语句返回数据, 标准用return
yield 返回数据后会保持挂起函数状态, 并会记住上次执行时的所有数据
目的: 方便每次在生成器调用__next__( ) 方法时, 从上次挂起的位置恢复继续执行 节省内存
生成器调用 返回一个生成器
"""
from typing import Iterator
def func():
    print(1)
    yield "a"
    print(2)
    yield "b"
    print(3)
    yield "c"    # 后面没的东西会输出 None
    print(45)

res = func()
print(func())    # 生成器函数在调用的时候返回一个地址 <generator object func at 0x000001B6B5A3EC10>
print(isinstance(res, Iterator))
print(res.__next__())   # 1 a 然后进入挂起状态 →  在下一次调用__next__时 会从挂起的状态继续 往后面执行, 直到再碰到下个 yield 为止
print(res.__next__())   # 2 b
print(res.__next__())   # 3 c 以此类推 __next__ 没有返回的值时会抛出错误

for i in res:    # 自动调用
    print(i)     # 1 a 2 b 3 c 45

print(list(res))   # 1 2 3 45 [a, b, c]  对生成器进行的迭代的时候, 是通过调用生成器函数, 获取里面的 yield 内容 , 所以print 内容也能输出

# 生成器逻辑
def gen():
    print("starting")
    for _ in range(2):
        print(res1 := (1, 2, 3, 4))
        res2 = yield 5, 6
        print(res2)
        res3 = yield res1
        print(res3)
        print(res1)

g4 = gen()
print(g4.__next__())    # starting, (1, 2, 3, 4), (5, 6)
print(g4.__next__())    # print(res2)的结果  None  在运行过后挂起, (1, 2, 3, 4)
print(g4.__next__())    # print(res3) = None (1, 2, 3, 4)

for i in g4:        # 抛出异常被 for 循环接收, 所以之后不会输出
    print(i)        # 把结果输出了两次def 里面的是 range(2) , (1, 2, 3, 4) (5, 6) None (1, 2, 3, 4) None (1, 2, 3, 4)

# 生成器表达式
print((sum(i for i in range(4))))   # 很像列表推导式

# 区别 一个小括号, 一个中括号
res1 = [i for i in range(6)]      # 列表推导式
print(sum(res1))     # 15
print(sum(res1))     # 15
res2 = (i for i in range(6))      # 生成器表达式
for i in res2:
    print(i)
print(sum(res2))     # 15
print(sum(res2))     # 0    原因 : 他也是迭代器, 已经迭代了6次, 所以从第六次开始 之后直接抛出错误 从0 开始 结果就是0

# iter(object, sentinel(标记, 哨兵)可不写)
"""
作用: 把一个对象转成迭代器, 由sentinel 决定停止  
返回一个迭代器对象
如果没有第二个参数, object 必须是一个可迭代对象
如果有第二个参数, object 必须是一个可调用的, 即函数, 方法, 类"""
res = iter([1, 2, 3, 4])
print(res)    # <list_iterator object at 0x000001B6B5B66970> 转成了一个列表迭代器 同理可转为其他可迭代对象

class A:
    def __call__(self):
        return 8

a = A()
res = iter(a, 9)     # 当迭代达到第二个值的时候会中断
print(iter(a, 9))    # 返回一个迭代器 <callable_iterator object at 0x000001B6B5D16F40>
print(res.__next__())   # 调用第一次得到8
print(res.__next__())   # 调用还是8  如果遍历还是 8 死循环

class A:
    def __init__(self, num):
        self.num = num
        
    def __call__(self):
        res = self.num
        self.num += 1
        return res

a = A(9)
res1 = iter(a, 20)
for i in res1:
    print(i)        # 输出 9 - 19  达到20立刻抛出错误 停止

# next(iterator, default)
""" 
作用: 获取里面的元素
通过调用迭代器的 __next__ 方法获取下一个元素, 迭代器耗尽 返回给定默认值, 如果没有默认值则抛出错误
"""
class A:
    def __init__(self, num):
        self.num = num
        
    def __call__(self):
        res = self.num
        self.num += 1
        return res

a = A(9)
res1 = iter(a, 20)   # 下面next 相当于一次取元素内容
print(next(res1))    # 9  相当于 res.__next__()
print(next(res1))    # 10 以此类推, 往下迭代 

"""
迭代器的优点: 提供不依赖索引的取值方式
内存释放: 节省内存, 迭代器只占一个数据的空间, 上一条数据会释放, 再加载此条数据, 不需要把所有数据加载到内存当中
缺点: 只能从头到尾, 不能跨, 不灵活
"""

# query 过滤帧筛选函数
"""
匹配过滤筛选符合条件的数据内容
"""
import pandas as pd
import numpy as np

values_1 = np.random.randint(10, size=10)
values_2 = np.random.randint(10, size=10)
years = np.arange(2010, 2020)
groups = ['A','A','B','A','B','B','C','A','C','C']
df = pd.DataFrame({'group':groups, 'year':years, 'value_1':values_1, 'value_2':values_2})
df1 = df.query('value_1 < value_2')     # 找出了值1小于值2的数据
print(df1)