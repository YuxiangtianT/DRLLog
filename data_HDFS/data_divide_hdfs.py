import random

# 输入文件路径
input_file_path = 'data/hdfs_test_abnormal'

# 输出文件路径
output_file_path = 'data/retrain_data/num/hdfs_retrain'

# # 截取的百分比（20%）
# percentage_to_extract = 4
#
# # 读取原始文件的所有行
# with open(input_file_path, 'r', encoding='utf-8') as input_file:
#     all_lines = input_file.readlines()
#
# # 计算要截取的行数
# total_lines = len(all_lines)
# lines_to_extract = int(total_lines * (percentage_to_extract / 100.0))
#
# # 随机选择要截取的行
# selected_lines = random.sample(all_lines, lines_to_extract)
#
# # 将选定的行写入输出文件
# with open(output_file_path + str(percentage_to_extract), 'w', encoding='utf-8') as output_file:
#     output_file.writelines(selected_lines)
#
# print(f"{lines_to_extract} lines extracted from {total_lines} total lines.")

# 截取的行数
lines_to_extract = 50

# 读取原始文件的所有行
with open(input_file_path, 'r', encoding='utf-8') as input_file:
    all_lines = input_file.readlines()

# 截取指定数量的行
selected_lines = all_lines[:lines_to_extract]

# 将选定的行写入输出文件
with open(output_file_path + str(lines_to_extract), 'w', encoding='utf-8') as output_file:
    output_file.writelines(selected_lines)

print(f"{lines_to_extract} lines extracted from the original file.")