import os
import pickle
import argparse
import pandas as pd
import numpy as np
# from utils import decision, json_pretty_dump
from collections import OrderedDict, defaultdict

parser = argparse.ArgumentParser()

parser.add_argument("--train_anomaly_ratio", default=0.0, type=float)

params = vars(parser.parse_args())

eval_name = f'bgl_{params["train_anomaly_ratio"]}_tar'
seed = 42
data_dir = "../data_BGL"
np.random.seed(seed)

params = {
    "log_file": "BGL.log_structured.csv",
    "session_size": 20,
    "train_ratio": None,
    "test_ratio": 0.2,
    "random_sessions": True,
    "train_anomaly_ratio": params["train_anomaly_ratio"],
}

data_dir = os.path.join(data_dir, eval_name)
os.makedirs(data_dir, exist_ok=True)


def load_BGL(
    log_file,
    session_size,
    train_ratio,
    test_ratio,
    random_sessions,
    train_anomaly_ratio,
):
    print("Loading BGL logs from {}.".format(log_file))
    struct_log = pd.read_csv(log_file, engine="c", na_filter=False, memory_map=True)
    # struct_log.sort_values(by=["Timestamp"], inplace=True)
    struct_log = struct_log.drop(["Date", "NodeRepeat"], axis=1)

    struct_log["Label"] = struct_log["Label"].map(lambda x: x != "-").astype(int).values
    struct_log["time"] = pd.to_datetime(
        struct_log["Time"], format="%Y-%m-%d-%H.%M.%S.%f"
    )
    session_dict = OrderedDict()
    column_idx = {col: idx for idx, col in enumerate(struct_log.columns)}
    for idx, row in enumerate(struct_log.values):
        current = idx
        if idx == 0:
            sessid = current
        elif current - sessid > session_size-1:
            sessid = current
        if sessid not in session_dict:
            session_dict[sessid] = defaultdict(list)
        session_dict[sessid]["EventIds"].append(row[column_idx["EventId"]])
        session_dict[sessid]["Label"].append(row[column_idx["Label"]])  # labeling for each log
        session_dict[sessid]["idx"].append(idx)

    normal_session_dict = OrderedDict()
    abnormal_session_dict = OrderedDict()
    # labeling for each session
    for k, v in session_dict.items():
        if 1 in v["Label"]:
            abnormal_session_dict[k]=v
        else:
            normal_session_dict[k]=v
    normal_session_idx = list(range(len(normal_session_dict)))
    # split data
    if random_sessions:
        print("Using random partition.")
        np.random.shuffle(normal_session_idx)

    normal_session_ids = np.array(list(normal_session_dict.keys()))

    if train_ratio is None:
        train_ratio = 1 - test_ratio
    train_lines = int(train_ratio * len(normal_session_idx))
    test_normal_lines = int(test_ratio * len(normal_session_idx))

    session_idx_train = normal_session_idx[0:train_lines]
    session_idx_test = normal_session_idx[-test_normal_lines:]

    session_id_train = normal_session_ids[session_idx_train]
    session_id_test = normal_session_ids[session_idx_test]

    print("Total # sessions: {}".format(len(session_dict)))

    session_train = {
        k: session_dict[k]
        for k in session_id_train
    }
    session_test_normal = {k: session_dict[k] for k in session_id_test}

    print("# train sessions: {} ({:.2f}%)".format(len(session_train), 0))
    print("# test sessions: {} ({:.2f}%)".format(len(session_dict)-len(session_train), 0.8))

    with open(os.path.join(data_dir, "session_train.pkl"), "wb") as fw:
        pickle.dump(session_train, fw)
    with open(os.path.join(data_dir, "session_test_normal.pkl"), "wb") as fw:
        pickle.dump(session_test_normal, fw)
    with open(os.path.join(data_dir, "session_test_abnormal.pkl"), "wb") as fw:
        pickle.dump(abnormal_session_dict, fw)
    # json_pretty_dump(params, os.path.join(data_dir, "data_desc.json"))
    print("Saved to {}".format(data_dir))


if __name__ == "__main__":
    load_BGL(**params)