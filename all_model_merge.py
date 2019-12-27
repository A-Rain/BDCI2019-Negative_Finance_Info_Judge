import pandas as pd
from utils import concat, data_path
import os


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


def count_wrong(x):
    count = 0
    for column in ["bert_ext_l2", "bert_ext_l2_pretrain", "bert_large_4v_lr2", "bert_large_4v_lr3",
                   "bert_large_4v_lr3_pretrain", "bert_large_4v_lr3_span"]:
        if x[column] > 0.5 and x["label"] == 0:
            count += 1
        if x[column] < 0.5 and x["label"] == 1:
            count += 1
    return count


def select(entity_list):
    new_list = [i for i in entity_list]
    #     不扔掉不出现在text中的实体
    #     for i in entity_list:
    #         if (i not in context) and (i not in title) and i not in good_entity:
    #             new_list.remove(i)
    new_list = sorted(new_list, key=lambda x: len(x), reverse=True)
    final_list = []
    for i in new_list:
        flag = True
        for j in final_list:
            if i in j:
                flag = False
                break
        if flag:
            final_list.append(i)
    return final_list


def drop_duplicates(data, submit_file):
    data['entity'] = data['entity'].fillna('')

    fold_no_sub = pd.DataFrame()
    for index, item in data.groupby('id'):
        entity_list = list(item['entity'])
        entity_list_new = select(entity_list)
        for index_t, item_t in item.iterrows():
            if item_t['entity'] in entity_list_new:
                fold_no_sub = fold_no_sub.append(item_t)

    fold_no_sub = pd.merge(fold_no_sub, submit_file[["id", "entity", "label"]], on=["id", "entity"], how="left")
    fold_no_sub["label"] = fold_no_sub["label"].apply(lambda x: 0 if x not in [0, 1] else x)
    fold_no_sub.drop_duplicates(inplace=True)
    return fold_no_sub


def average():
    same_columns = ["id", "text", "entity"]
    probs_columns = ["bert_ext_l2", "bert_ext_l2_pretrain", "bert_large_4v_lr2", "bert_large_4v_lr3",
                     "bert_large_4v_lr3_pretrain", "bert_large_4v_lr3_span"]

    cv_0 = pd.read_csv("../data/cv/all_cv_0_nocls.csv", encoding='utf-8')[same_columns + probs_columns]
    cv_1 = pd.read_csv("../data/cv/all_cv_1_nocls.csv", encoding='utf-8')[same_columns + probs_columns]
    cv_2 = pd.read_csv("../data/cv/all_cv_2_nocls.csv", encoding='utf-8')[same_columns + probs_columns]
    cv_3 = pd.read_csv("../data/cv/all_cv_3_nocls.csv", encoding='utf-8')[same_columns + probs_columns]
    cv_4 = pd.read_csv("../data/cv/all_cv_4_nocls.csv", encoding='utf-8')[same_columns + probs_columns]

    cv_0.drop_duplicates(inplace=True)
    cv_1.drop_duplicates(inplace=True)
    cv_2.drop_duplicates(inplace=True)
    cv_3.drop_duplicates(inplace=True)
    cv_4.drop_duplicates(inplace=True)

    cv0_rename = {column: column + "_cv0" for column in probs_columns}
    cv1_rename = {column: column + "_cv1" for column in probs_columns}
    cv2_rename = {column: column + "_cv2" for column in probs_columns}
    cv3_rename = {column: column + "_cv3" for column in probs_columns}
    cv4_rename = {column: column + "_cv4" for column in probs_columns}

    cv_0.rename(columns=cv0_rename, inplace=True)
    cv_1.rename(columns=cv1_rename, inplace=True)
    cv_2.rename(columns=cv2_rename, inplace=True)
    cv_3.rename(columns=cv3_rename, inplace=True)
    cv_4.rename(columns=cv4_rename, inplace=True)

    data = pd.merge(cv_0, cv_1, on=same_columns, how="left")
    data = pd.merge(data, cv_2, on=same_columns, how="left")
    data = pd.merge(data, cv_3, on=same_columns, how="left")
    data = pd.merge(data, cv_4, on=same_columns, how="left")

    for column in probs_columns:
        data[column] = data.apply(lambda x: (x[column + "_cv0"] + x[column + "_cv1"] + x[column + "_cv2"]
                                             + x[column + "_cv3"] + x[column + "_cv4"]) / 5, axis=1)
    return data[same_columns + probs_columns]


def compare_less(value_list):
    import itertools
    for i in itertools.combinations(value_list, 5):
        counter = 0
        for value in i:
            if float('%.2f' % value) <= 0.4:
                counter += 1
        if counter == 5:
            return counter
    return 0


def compare_more(value_list):
    import itertools
    for i in itertools.combinations(value_list, 5):
        counter = 0
        for value in i:
            if float('%.2f' % value) >= 0.6:
                counter += 1
        if counter == 5:
            return counter
    return 0


def update(x, data):
    if type(x["key_entity"]) is float:
        return ""
    key_entity_list = x["key_entity"].split(';')
    entity_list = x["entity"].split(';')

    for idx, item in data.iterrows():
        if item["count_wrong"] == 6 and item["cos_similar"] < 0.7 and item["cos_similar_v2"] < 0.7:
            if item["label"] == 0 and item["bert_ext_l2"] > 0.6 and item["bert_ext_l2_pretrain"] > 0.6 and \
                item["bert_large_4v_lr2"] > 0.6 and item["bert_large_4v_lr3"] > 0.6 and \
                item["bert_large_4v_lr3_pretrain"] > 0.6 and item["bert_large_4v_lr3_span"] > 0.6:
                if item["entity"] not in item["text"]:
                    for e in entity_list:
                        if e in item["entity"] and e in item["text"]:
                            key_entity_list.append(e)
                else:
                    key_entity_list.append(item["entity"])
            if item["label"] == 1 and item["bert_ext_l2"] < 0.4 and item["bert_ext_l2_pretrain"] < 0.4 and \
                item["bert_large_4v_lr2"] < 0.4 and item["bert_large_4v_lr3"] < 0.4 and \
                item["bert_large_4v_lr3_pretrain"] < 0.4 and item["bert_large_4v_lr3_span"] < 0.4:
                key_entity_list.remove(item["entity"])

        if item["count_wrong"] == 5 and item["cos_similar"] < 0.7 and item["cos_similar_v2"] < 0.7:
            if item["label"] == 0:
                counter = compare_more([item["bert_ext_l2"], item["bert_ext_l2_pretrain"], item["bert_large_4v_lr2"],
                              item["bert_large_4v_lr3"], item["bert_large_4v_lr3_pretrain"],
                              item["bert_large_4v_lr3_span"]])
                if counter == 5:
                    if item["entity"] not in item["text"]:
                        for e in entity_list:
                            if e in item["entity"] and e in item["text"]:
                                key_entity_list.append(e)
                    else:
                        key_entity_list.append(item["entity"])
            elif item["label"] == 1:
                counter = compare_less([item["bert_ext_l2"], item["bert_ext_l2_pretrain"], item["bert_large_4v_lr2"],
                         item["bert_large_4v_lr3"], item["bert_large_4v_lr3_pretrain"], item["bert_large_4v_lr3_span"]])
                if counter == 5:
                    if item["entity"] not in item["text"]:
                        for e in entity_list:
                            if e in item["entity"] and e in item["text"]:
                                key_entity_list.remove(e)
                    else:
                        key_entity_list.remove(item["entity"])
    return ";".join(key_entity_list)


def update_submit(data):
    test_data = pd.read_csv(os.path.join(data_path, "preprocess", "Test_Data_round2.csv"), encoding='utf-8')
    submit_data = pd.read_csv(os.path.join(data_path, "submit", "fuxian_replace.csv"), encoding='utf-8')
    submit_data = pd.merge(test_data, submit_data, on="id")
    submit_data["key_entity"] = submit_data.apply(lambda x: update(x, data[data["id"] == x["id"]]), axis=1)
    submit_data["negative"] = submit_data["key_entity"].apply(lambda x: 0 if type(x) is float or x == "" else 1)
    submit_data[["id", "negative", "key_entity"]].to_csv(os.path.join(data_path, "submit", "fuxian_replace_post.csv"),
                                                         encoding='utf-8', index=False)


if __name__ == "__main__":
    test_data = pd.read_csv(os.path.join(data_path, "preprocess", "Test_Data_round2.csv"), encoding='utf-8')
    submit_data = pd.read_csv(os.path.join(data_path, "submit", "fuxian_replace.csv"), encoding='utf-8')
    submit_data = pd.merge(test_data, submit_data, on="id")
    submit_data = get_entity_data(submit_data)

    data = average()
    data = drop_duplicates(data, submit_data)

    data["count_wrong"] = data.apply(lambda x: count_wrong(x), axis=1)

    test_cos = pd.read_csv(os.path.join(data_path, "test_cos_text.csv"), encoding='utf-8')
    test_cos_v2 = pd.read_csv(os.path.join(data_path, "test_cos_text_v2.csv"), encoding='utf-8')
    test_cos_v2.rename(columns={"cos_similar": "cos_similar_v2"}, inplace=True)

    data = pd.merge(data, test_cos[["id", "cos_similar"]], on="id", how="left")
    data = pd.merge(data, test_cos_v2[["id", "cos_similar_v2"]], on="id", how="left")

    data[["id", "text", "entity", "label", "bert_ext_l2", "bert_ext_l2_pretrain", "bert_large_4v_lr2",
          "bert_large_4v_lr3", "bert_large_4v_lr3_pretrain", "bert_large_4v_lr3_span", "count_wrong",
          "cos_similar", "cos_similar_v2"
          ]].to_csv(os.path.join(data_path, "cv", "new_model_average_drop_wrong.csv"), encoding='utf-8-sig', index=False)

    update_submit(data)
