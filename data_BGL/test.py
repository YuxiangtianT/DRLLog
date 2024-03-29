import pickle

# 打开.pkl文件以二进制读取模式打开
with open('wcl/session_train410.pkl', 'rb') as file:
    # 使用pickle.load()加载.pkl文件中的对象
    data = pickle.load(file)

print(data)

# lines = []
# for key, _ in data.items():
#     lines.append(data[key]['EventIds'])
# print(len(lines))

# for line in lines:
#     print(line)

# 现在，'data'变量中包含了.pkl文件中的对象
# print(data)