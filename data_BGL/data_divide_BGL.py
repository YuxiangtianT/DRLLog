import pickle

# 打开.pkl文件以二进制读取模式打开
with open('data_BGL/bgl_0.0_tar/session_train.pkl', 'rb') as file:
    # 使用pickle.load()加载.pkl文件中的对象
    data = pickle.load(file)

# 获取字典的键
keys = list(data.keys())

# 计算分割点，这里按照7:1的比例进行划分
split_point = int(len(keys) * 7 / 8)

# 划分字典
dict_part1 = {key: data[key] for key in keys[:split_point]}
dict_part2 = {key: data[key] for key in keys[split_point:]}

# 将划分的两部分字典存储到新的文件中
with open('data_BGL/retrain_data/percent/71/session_train.pkl', 'wb') as file_part1:
    pickle.dump(dict_part1, file_part1)

with open('data_BGL/retrain_data/percent/71/session_retrain.pkl', 'wb') as file_part2:
    pickle.dump(dict_part2, file_part2)
