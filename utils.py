import pandas as pd
import os


data_path = "../data"
train_path = os.path.join(data_path, "origin_data", "Round2_train.csv")
test_path = os.path.join(data_path, "origin_data", "round2_test.csv")
device_id = 0


def load_data():
    train_data = pd.read_csv(os.path.join(train_path), encoding='utf-8')
    test_data = pd.read_csv(os.path.join(test_path), encoding='utf-8')
    return train_data, test_data


def concat(title, content):
    if type(title) is float or str(title) in str(content):
        return str(content)
    else:
        return str(title) + " " + str(content)


def entity_in_text_counter(data):
    entity_set = []
    entity_counter = {}
    for i in range(len(data)):
        entity = data.iloc[i]["entity"]
        entity_set.append(entity)
    for entity in set(entity_set):
        entity_counter[entity] = entity_set.count(entity)
    return entity_counter


def entity_not_in_text_counter(data):
    entity_set = []
    entity_counter = {}
    for i in range(len(data)):
        entity = data.iloc[i]["entity"]
        if entity not in data.iloc[i]["text"]:
            entity_set.append(entity)
    for entity in set(entity_set):
        entity_counter[entity] = entity_set.count(entity)
    return entity_counter
