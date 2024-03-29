# with open('test_normal', 'r') as f:
#     lines = f.readlines()
#
# print(len(lines))
#
# num = 0
#
# for line in lines:
#     num += len(line.strip().split(" "))
#
# print(num)

# import pandas as pd
#
# # 读取 CSV 文件
# df = pd.read_csv("openstack_result/openstack_normal2.log_structured.csv")
#
# # 提取 EventId 列
# event_ids = df['EventId']
#
# # 将 EventId 分成 20 个一组
# grouped_event_ids = [event_ids[i:i+20].tolist() for i in range(0, len(event_ids), 20)]
#
# # 将结果保存到 txt 文件
# with open("openstack_test_normal", "w") as file:
#     for group in grouped_event_ids:
#         line = " ".join(map(str, group))
#         file.write(line + "\n")

