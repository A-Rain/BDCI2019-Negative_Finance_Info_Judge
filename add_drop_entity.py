import os
import pandas as pd
from utils import concat, data_path, train_path, load_data


def get_entity_data(data):
    data = data[data["negative"] == 1]
    data["text"] = data.apply(lambda x: concat(x['title'], x['text']), axis=1)

    id = []
    text = []
    entities = []
    label = []
    for i in range(len(data)):
        entity = data["entity"].iloc[i].split(';')
        try:
            key_entity = data["key_entity"].iloc[i].split(';')
        except Exception as error:
            key_entity = []
        for e in entity:
            if e is "":
                continue
            id.append(data["id"].iloc[i])
            text.append(data["text"].iloc[i])
            entities.append(e)
            if e in key_entity:
                label.append(1)
            else:
                label.append(0)
    entity_data = pd.DataFrame({"id": id, "text": text, "entity": entities, "label": label})

    return entity_data


def not_in_text(x):
    entity = x["entity"]
    text = x["text"]
    if entity not in text:
        return 1
    else:
        return 0


def add_entity(x, negative_entity_list):
    if x["flag"] != 1:
        return x["label"]
    else:
        if x["entity"] in negative_entity_list:
            return 1
        else:
            return x["label"]


def drop_entity(x, positive_entity_list):
    if x["flag"] != 1:
        return x["label"]
    else:
        if x["entity"] in positive_entity_list:
            return 0
        else:
            return x["label"]


def to_submit_format(data, test_data, file=""):
    submit = {}
    for i in range(len(data)):
        id = data.iloc[i]["id"]
        entity = data.iloc[i]["entity"]
        label = data.iloc[i]["label"]
        if id not in submit.keys():
            submit[id] = []
        if label == 1:
            submit[id].append((entity, 1))
        else:
            submit[id].append((entity, 0))
    submit = pd.DataFrame({"id": list(submit.keys()), "entity": list(submit.values())})
    submit["key_entity"] = submit["entity"].apply(lambda x: "" if len(x) == 0 else ';'.join(set([_ for _, l in x if l == 1])))
    submit = pd.merge(test_data[["id", "negative"]], submit[["id", "key_entity"]], on="id", how="left")
    if file:
        submit[["id", "negative", "key_entity"]].to_csv(os.path.join(data_path, "submit", file), encoding='utf-8', index=False)


if __name__ == "__main__":
    train_data, test_data = load_data()
    submit_data = pd.read_csv(os.path.join(data_path, "submit", "fuxian_result.csv"), encoding='utf-8')
    test_data = pd.merge(test_data, submit_data, on="id")
    train_data = get_entity_data(train_data)
    test_data = get_entity_data(test_data)

    train_data["flag"] = train_data.apply(lambda x: not_in_text(x), axis=1)
    train_data[train_data["flag"] == 1].to_csv(os.path.join(data_path, "entity_train_not_in_text.csv"),
                                               encoding='utf-8-sig', index=False)

    test_data["flag"] = test_data.apply(lambda x: not_in_text(x), axis=1)

    entity_counter = {}
    negative_counter = {}
    positive_counter = {}
    train_data = pd.read_csv(os.path.join(data_path, "entity_train_not_in_text.csv"), encoding='utf-8-sig')

    for i in range(len(train_data)):
        entity_counter[train_data.iloc[i]["entity"]] = entity_counter.get(train_data.iloc[i]["entity"], 0) + 1
        if train_data.iloc[i]["label"] == 0:
            positive_counter[train_data.iloc[i]["entity"]] = positive_counter.get(train_data.iloc[i]["entity"], 0) + 1
        else:
            negative_counter[train_data.iloc[i]["entity"]] = negative_counter.get(train_data.iloc[i]["entity"], 0) + 1

    for key, value in negative_counter.items():
        negative_counter[key] = negative_counter[key] / entity_counter[key]

    for key, value in positive_counter.items():
        positive_counter[key] = positive_counter[key] / entity_counter[key]

    negative_entity_list = [entity for entity, _ in negative_counter.items() if _ == 1.0]
    positive_entity_list = [entity for entity, _ in positive_counter.items() if _ == 1.0]

    test_data["label"] = test_data.apply(lambda x: add_entity(x, negative_entity_list), axis=1)
    test_data["label"] = test_data.apply(lambda x: drop_entity(x, positive_entity_list), axis=1)

    to_submit_format(test_data, submit_data, "fuxian_add_drop.csv")
