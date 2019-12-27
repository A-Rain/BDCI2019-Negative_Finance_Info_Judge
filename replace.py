import pandas as pd
import os
from utils import data_path


def drop_duplicate_key_entity(test_list, train_list):
    if type(test_list) is float and type(train_list) is float:
        return 1
    if type(test_list) is not float and type(train_list) is not float:
        train_list = train_list.split(';')
        test_list = test_list.split(';')
        if set(train_list) != set(test_list):
            return 0
        else:
            return 1
    else:
        return 0


def get_id_map(data, cos_similar):
    data = data[data["cos_similar"] > cos_similar]
    data["same"] = data.apply(lambda x: 1 if x["entity"] == x["similar_entity"] else 0, axis=1)
    data = data[data["same"] == 1].reset_index(drop=True)
    del data["same"]
    data["same"] = data.apply(lambda x: drop_duplicate_key_entity(x["key_entity"], x["similar_key_entity"]), axis=1)
    data = data[data["same"] == 0]
    del data["same"]
    id_map = {}
    for i in range(len(data)):
        if len(data.iloc[i]["text"]) > 25:
            id_map[data.iloc[i]["id"]] = data.iloc[i]["similar_id"]
    data = data[data["id"].isin(id_map)]
    print(data)
    return id_map


def replace_negative(id_map, id, negative, train_data):
    if id in id_map:
        train_id = id_map[id]
        return train_data[train_data["id"] == train_id]["negative"].values[0]
    else:
        return negative


def replace_key_entity(id_map, id, key_entity, train_data):
    if id in id_map:
        train_id = id_map[id]
        return train_data[train_data["id"] == train_id]["key_entity"].values[0]
    else:
        return key_entity


def change_replace(file, cos_file):
    submit = pd.read_csv(os.path.join(data_path, "submit", file))
    test_data = pd.read_csv(os.path.join(data_path, "preprocess", "Test_Data_round2.csv"), encoding='utf-8')
    test_data = pd.merge(test_data[["id", "entity"]], submit, on="id", how="left")
    data = pd.read_csv(os.path.join(data_path, cos_file), encoding='utf-8')
    del data["negative"]
    del data["entity"]
    del data["key_entity"]
    data = pd.merge(data, test_data, on="id", how="left")
    # data[["id", "text", "similar_text", "cos_similar", "similar_id", "category", "entity", "negative",
    #       "key_entity", "similar_negative", "similar_entity", "similar_key_entity"]].\
    #     to_csv(os.path.join(data_path, "test_cos_best.csv"), encoding='utf-8', index=False)
    return data[["id", "text", "similar_text", "cos_similar", "similar_id", "category", "entity", "negative",
                 "key_entity", "similar_negative", "similar_entity", "similar_key_entity"]]


def replace_train(cos_file, cos_similar, file=""):
    submit = pd.read_csv(os.path.join(data_path, "submit", file))
    train_data = pd.read_csv(os.path.join(data_path, "preprocess", "Train_Data_round2.csv"), encoding='utf-8')
    id_map = get_id_map(cos_file, cos_similar)
    print(len(id_map))
    print(submit[submit["id"].isin(id_map)])
    submit["negative"] = submit.apply(
        lambda x: replace_negative(id_map, x["id"], x["negative"], train_data), axis=1)
    submit["key_entity"] = submit.apply(
        lambda x: replace_key_entity(id_map, x["id"], x["key_entity"], train_data), axis=1)
    return submit


if __name__ == "__main__":

    cos_file = change_replace("fuxian_add_drop_dundoukong_substring_1.csv", "test_cos_text_v2.csv")
    submit = replace_train(cos_file, 0.68, "fuxian_add_drop_dundoukong_substring_1.csv")
    submit.to_csv(os.path.join(data_path, "submit", "fuxian_replace.csv"), encoding='utf-8', index=False)

    cos_file = change_replace("fuxian_replace.csv", "test_cos_text.csv")
    submit = replace_train(cos_file, 0.7, "fuxian_replace.csv")
    submit.to_csv(os.path.join(data_path, "submit", "fuxian_replace.csv"), encoding='utf-8', index=False)


