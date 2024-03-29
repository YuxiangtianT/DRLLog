import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import pickle
from gensim.models import word2vec


# 定义深度神经网络模型
# class DQN(nn.Module):
# def __init__(self, input_size, action_size):
#     super(DQN, self).__init__()
#     self.fc1 = nn.Linear(input_size, 128)
#     self.relu = nn.ReLU()
#     self.fc2 = nn.Linear(128, 128)
#     self.fc3 = nn.Linear(128, action_size)
#
# def forward(self, x):
#     x = self.fc1(x)
#     x = self.relu(x)
#     x = self.fc2(x)
#     x = self.relu(x)
#     x = self.fc3(x)
#     return x

# def __init__(self, input_size, action_size):
#     super(DQN, self).__init__()
#     self.model = nn.Sequential(
#         nn.Linear(input_size, 128),
#         nn.ReLU(),
#         nn.Linear(128, action_size)
#     )
#
# def forward(self, state):
#     return self.model(state)


class DQN(nn.Module):
    def __init__(self, input_size, action_size):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, state):
        return self.model(state)


# 定义经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )


def huber_loss(predictions, targets, delta=1.0):
    errors = torch.abs(predictions - targets)
    condition = errors < delta
    loss = torch.where(condition, 0.5 * errors ** 2, delta * (errors - 0.5 * delta))
    return torch.mean(loss)


def log_cosh_loss(predictions, targets):
    errors = predictions - targets
    loss = torch.log(torch.cosh(errors))
    return torch.mean(loss)


# 定义DQNAgent
class DQNAgent:
    def __init__(self, input_size, action_size):
        self.input_size = input_size
        self.action_size = action_size
        self.q_network = DQN(input_size, action_size)
        self.target_q_network = DQN(input_size, action_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)  # 默认1e-3
        self.gamma = 0.99
        self.epsilon = 0.1
        self.replay_buffer = ReplayBuffer(capacity=1000000)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state = torch.FloatTensor(state)
            with torch.no_grad():
                q_values = self.q_network(state)
            return q_values.argmax().item()

    def train(self, epoch, batch_size=2048):
        if len(self.replay_buffer.buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_q_network(next_states).max(dim=1, keepdim=True)[0].detach()
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.functional.mse_loss(q_values, targets)
        # loss = huber_loss(q_values, targets)
        # loss = log_cosh_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

    def update_target_network(self, epoch, save_file_path):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        # 保存 target_q_network 模型参数到文件
        if epoch == num_epoch:
            save_path = os.path.join(save_file_path, f'target_q_network_epoch{epoch}_action1_r50.pt')
            torch.save(self.target_q_network.state_dict(), save_path)

    def load_model(self, retrain_model_path):
        self.q_network.load_state_dict(torch.load(retrain_model_path), strict=False)
        self.target_q_network.load_state_dict(self.q_network.state_dict())


# 环境模拟（文本日志）
# class DummyTextEnvironment:
#     def __init__(self):
# self.log_data = [
#     "error: unable to connect to database",
#     "warning: high CPU usage",
#     "info: system rebooted successfully",
#     "error: application crashed"
# ]
# self.log_data = []
# self.input_size = 1
# self.action_space = 2
# self.vectorizer = CountVectorizer()

# def _get_state(self):
#     return self.vectorizer.transform(self.log_data).toarray().flatten()

# def reset(self):
#     self.log_data = []
# return self._get_state()

# def step(self, action):
#     new_log = "info: normal operation"
#     self.log_data.append(new_log)
#     next_state = self.vectorizer.fit_transform(self.log_data).toarray().flatten()
#     reward = 1.0 if "error" not in new_log else -1.0
#     done = False
#     return next_state, reward, done

# def step(self, new_log):
#     self.log_data.append(new_log)
#     next_state = self._get_state()
#     reward = 1.0
#     down = False
#     return next_state, reward, down


def encode_one_hot(log_id, num_classes):
    encoded_vector = np.zeros(num_classes)
    encoded_vector[log_id - 1] = 1
    return encoded_vector


# def process_file(file_path, lines_per_chunk):
#     with open(file_path, 'r') as f:
#         lines = []
#         total_reward = 0
#         for line in f:
#             lines.append(line)
#             if len(lines) == lines_per_chunk:
#                 process_lines(lines)
#                 agent.update_target_network()
#                 # print(f"Epoch {epoch}, Total Reward: {total_reward}")
#                 lines = []
#                 total_reward = 0
#
#         # 处理最后一批不足 1000 行的行
#         if lines:
#             process_lines(lines)
#             agent.update_target_network()
#             # print(f"Epoch {epoch}, Total Reward: {total_reward}")

# def process_lines(lines):
#     # 在这里进行对每行的处理
#     for line in lines:
#         total_reward = 0
#         entries = [int(entry) for entry in line.strip().split(' ')]
#         for i in range(len(entries) - 1):
#             state = entries[i]
#             state_vector = encode_one_hot(state, num_classes)
#             action = agent.select_action(state_vector)
#             # print(action)
#             next_state = entries[i + 1]
#             next_state_vector = encode_one_hot(next_state, num_classes)
#             if i == len(entries) - 2:
#                 reward = 100.0
#                 # final_state.append(state)
#                 done = True
#             else:
#                 reward = 1.0
#                 done = False
#             agent.replay_buffer.push(state_vector, action, reward, next_state_vector, done)
#             agent.train()
#             state = next_state
#             total_reward += reward


def open_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return lines

def open_file_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    lines = []
    for key, _ in data.items():
        lines.append(data[key]['EventIds'])

    return lines

def train(file_path, num_epoch, save_file_path):
    # lines = open_file(file_path)
    lines = open_file_pkl(file_path)
    # vec_model = word2vec.Word2Vec.load('data/w2c.model')
    for line in lines:
        total_reward = 0
        entries = line
        # entries = [int(entry) for entry in line.strip().split(' ')]
        # entries = line.strip().split(' ')
        for i in range(len(entries) - 1):
            state = entries[i]
            state_vector = encode_one_hot(state, num_classes)
            # state_vector = vec_model.wv[state]
            action = agent.select_action(state_vector)
            # print(action)
            next_state = entries[i + 1]
            next_state_vector = encode_one_hot(next_state, num_classes)
            # next_state_vector = vec_model.wv[next_state]
            if i == len(entries) - 2:
                reward = 50.0
                # final_state.append(state)
                done = True
                final_states_set.add(next_state)
            else:
                reward = 1.0
                done = False
            agent.replay_buffer.push(state_vector, action, reward, next_state_vector, done)
            state = next_state
            total_reward += reward
            # print(total_reward)

    with open('data_BGL/final_states_set.pkl', 'wb') as f:
        pickle.dump(final_states_set, f)

    for epoch in range(1, num_epoch + 1):
        agent.train(epoch)
        agent.update_target_network(epoch, save_file_path)
        # print(f"Epoch: {epoch}, Total Reward: {epoch}")


def predict(model_path, input_size, action_size, num_classes, threshold_1):
    model = DQN(input_size, action_size)
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()

    # vec_model = word2vec.Word2Vec.load('data/w2c.model')

    # normal_lines = open_file('data/hdfs_test_normal')
    # abnormal_lines = open_file('data/hdfs_test_abnormal')

    normal_lines = open_file_pkl('data_BGL/wcl/session_test_normal410.pkl')
    abnormal_lines = open_file_pkl('data_BGL/wcl/session_test_abnormal410.pkl')

    # normal_lines = open_file('data_OpenStack/openstack_test_normal')
    # abnormal_lines = open_file('data_OpenStack/openstack_test_abnormal')

    TP = 0
    FP = 0

    # 从文件中加载 final_states_set
    with open('data_BGL/final_states_set.pkl', 'rb') as f:
        final_states_set = pickle.load(f)

    with torch.no_grad():
        for line in normal_lines:
            flag = False
            entries = line
            # entries = [int(entry) for entry in line.strip().split(' ')]
            # entries = line.strip().split(' ')
            for i in range(len(entries) - 1):
                state = entries[i]
                state_vector = encode_one_hot(state, num_classes)
                # state_vector = vec_model.wv[state]
                state_vector = torch.FloatTensor(state_vector)
                q_value = model(state_vector)[0]
                # print(q_value)
                if q_value < threshold_1:
                    FP += 1
                    flag = True
                    break

            # 判断最后一个条目是否在最终状态中
            if not flag:
                if entries[len(entries) - 1] not in final_states_set:
                    FP += 1

    with torch.no_grad():
        for line in abnormal_lines:
            flag = False
            entries = line
            # entries = [int(entry) for entry in line.strip().split(' ')]
            # entries = line.strip().split(' ')
            for i in range(len(entries) - 1):
                state = entries[i]
                state_vector = encode_one_hot(state, num_classes)
                # state_vector = vec_model.wv[state]
                state_vector = torch.FloatTensor(state_vector)
                q_value = model(state_vector)
                # q_value = torch.round(q_value * 100) / 100
                # print(q_value)
                if q_value < threshold_1:
                    TP += 1
                    flag = True
                    break

            # 判断最后一个条目是否在最终状态中
            if not flag:
                if entries[len(entries) - 1] not in final_states_set:
                    TP += 1

    FN = len(abnormal_lines) - TP

    # print(FN)
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print(
        'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(
            FP, FN, P, R, F1))
    print('Finished Predicting')


def retrain(retrain_model_path, retrain_file_path, num_epoch, save_file_path):
    agent.load_model(retrain_model_path)
    # lines = open_file(retrain_file_path)
    lines = open_file_pkl(retrain_file_path)
    for line in lines:
        entries = line
        # entries = [int(entry) for entry in line.strip().split(' ')]
        for i in range(len(entries) - 1):
            state = entries[i]
            state_vector = encode_one_hot(state, num_classes)
            action = agent.select_action(state_vector)
            next_state = entries[i + 1]
            next_state_vector = encode_one_hot(next_state, num_classes)
            # if i == len(entries) - 2:
            #     reward = 50.0
            #     done = True
            #     final_states_set.add(next_state)
            # else:
            #     reward = 1.0
            #     done = False
            reward = -1.0
            done = False
            agent.replay_buffer.push(state_vector, action, reward, next_state_vector, done)
            state = next_state

    for epoch in range(1, num_epoch + 1):
        agent.train(epoch)
        agent.update_target_network(epoch, save_file_path)



# 训练DQN（文本日志）
# text_env = DummyTextEnvironment()
# agent = DQNAgent(input_size=len(text_env.reset()), action_size=text_env.action_space)


if __name__ == '__main__':
    # ile_path = 'data/hdfs_train'
    file_path = 'data_BGL/wcl/session_train410.pkl'
    # lines_per_chunk = 1000
    input_size = 410 # 28 156 2654
    action_size = 1
    num_classes = 410 # 28 156 2654
    num_epoch = 1000
    final_states_set = set()
    save_file_path = 'model_BGL/2024.1.10'
    model_path = 'model_BGL/2024.1.10/target_q_network_epoch1000_action1_r50.pt'
    threshold_1 = 50

    retrain_num_epoch = 1000
    retrain_model_path = 'model_BGL/2023.12.28/target_q_network_epoch1000_action1_r50_loss_data71.pt'
    retrain_file_path = 'data_BGL/retrain_data/percent/71/retrain.pkl'
    retrain_save_file_path = 'model_BGL/2023.12.28/retrain'
    retrain_threshold_1 = 0

    agent = DQNAgent(input_size=input_size, action_size=action_size)

    # for epoch in range(num_epoch):
    #     process_file(file_path, lines_per_chunk)

    # train(file_path, num_epoch, save_file_path)

    predict(model_path, input_size, action_size, num_classes, threshold_1)

    # retrain(retrain_model_path, retrain_file_path, retrain_num_epoch, retrain_save_file_path)

    retrain_predict_model_path = 'model_BGL/2023.12.28/retrain/target_q_network_epoch1000_action1_r50_loss_data71.pt'

    # predict(retrain_predict_model_path, input_size, action_size, num_classes, retrain_threshold_1)
