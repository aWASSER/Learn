from string import punctuation
import pandas as pd
from tqdm import tqdm
import jieba

punctuation = r""",！，。^%$@%^&()*[\]\#\"{|}~![]【】:;?!\'"""

def is_punctuation(ch):
    return ch in punctuation

def split_text(text):
    """
    针对给定文本进行分词操作
    : param text: 待分词文本
    : return: 分词列表
    """
    words = []
    for word in jieba.cut(text):
        word = word.strip()
        if len(word) == 0:
            continue
        if is_punctuation(text):        # 将所有的标点符号换成特殊值
            word = "E"
        words.append(word)
    return words

def train():
    path = r'../datas/文件.csv'
    # 数据加载
    df = pd.read_csv(path)
    print(df.head(10))
    # 分词
    y = []
    x0 = []
    with open('./t0.txt', 'w', encoding='utf-8') as writer:
        for value in tqdm(df.values):
            y.append(int(value[0]))
            _words = split_text(str(value[1]))
            x0.append(_words)
            writer.writelines(f"{value[0]}, {value[1]}, {'|'.join(_words)}\n")
    pass






if __name__ == "__main__":
    "流程--->加载数据--->构建模型--->模型训练--->模型保存"
    pass