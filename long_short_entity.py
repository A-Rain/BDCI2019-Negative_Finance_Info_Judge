import os
import pandas as pd
from utils import data_path


def get_keep_pair_dict():
    long_short_count_dict = {}

    for index, cur_row in train_data.iterrows():
        if type(cur_row['entity']) == float:
            continue

        entity_list = cur_row['entity'].split(';')
        if '' in entity_list:
            entity_list.remove('')
        entity_list_processed = list(set(entity_list))

        if type(cur_row['key_entity']) == float:
            continue
        key_entity_list = cur_row['key_entity'].split(';')
        if '' in key_entity_list:
            key_entity_list.remove('')
        key_entity_processed = list(set(key_entity_list))

        # entity_outer 是短实体
        for entity_outer in entity_list_processed:
            # entity_inner 是长实体
            for entity_inner in entity_list_processed:
                if entity_outer == entity_inner:
                    continue
                else:
                    if entity_outer in entity_inner:
                        entity_tupple = (entity_outer, entity_inner)
                        if entity_tupple not in long_short_count_dict:
                            long_short_count_dict[entity_tupple] = {'both': 0, 'only_longer': 0, 'only_short': 0, 'no_one': 0}

                        if entity_outer in key_entity_processed and entity_inner in key_entity_processed:
                            long_short_count_dict[entity_tupple]['both'] += 1
                        elif entity_outer in key_entity_processed and entity_inner not in key_entity_processed:
                            long_short_count_dict[entity_tupple]['only_short'] += 1
                        elif entity_outer not in key_entity_processed and entity_inner in key_entity_processed:
                            long_short_count_dict[entity_tupple]['only_longer'] += 1
                        else:
                            long_short_count_dict[entity_tupple]['no_one'] += 1

    keep_pair_dict = {}

    count = 0
    for key, val in long_short_count_dict.items():
        # 将符合筛选条件的实体放到字典中
        if (val['both'] >= 1 and val['only_longer'] == 0) or (val['both'] >= 3 and val['only_longer'] == 1):
            keep_pair_dict[key] = val
            count += 1
            print(key)
            print(val)
            print('\n')
    print(count)
    return keep_pair_dict


def post_test_data(keep_pair_dict):
    change_count = 0
    for index, cur_row in merge_data.iterrows():
        if type(cur_row['entity']) == float:
            continue
        if type(cur_row['key_entity']) == float:
            continue

        entity_list = list(set(cur_row['entity'].split(';')))
        if '' in entity_list:
            entity_list.remove('')

        key_entity_list = list(set(cur_row['key_entity'].split(';')))
        if '' in key_entity_list:
            key_entity_list.remove('')

        need_add_entity_list = []
        # entity_outer 是短实体
        for entity_outer in entity_list:
            # entity_inner 是长实体
            for entity_inner in entity_list:
                if entity_outer == entity_inner:
                    continue
                else:
                    # entity_list 里既出现了短串，也出现了长串
                    if entity_outer in entity_inner:
                        short_long_entity_pair = (entity_outer, entity_inner)

                        if short_long_entity_pair in keep_pair_dict:
                            if entity_outer in key_entity_list and entity_inner not in key_entity_list:
                                need_add_entity_list.append(entity_inner)
                                change_count += 1
                            elif entity_outer not in key_entity_list and entity_inner in key_entity_list:
                                need_add_entity_list.append(entity_outer)
                                change_count += 1
        if len(need_add_entity_list) != 0:
            new_key_entity_list = list(set(key_entity_list + need_add_entity_list))
            print(key_entity_list)
            merge_data.loc[index:index, 'key_entity'] = ';'.join(new_key_entity_list)
            print(new_key_entity_list)

    print('增加的实体个数为：{}'.format(change_count))


if __name__ == "__main__":
    train_path = os.path.join(data_path, "preprocess", "Train_Data_round2.csv")
    test_path = os.path.join(data_path, "preprocess", "Test_Data_round2.csv")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    model_predict_result = pd.read_csv(os.path.join(data_path, "submit", "fuxian_add_drop_dundoukong.csv"))
    merge_data = test_data.merge(model_predict_result, left_on='id', right_on='id')

    keep_pair_dict = get_keep_pair_dict()
    post_test_data(keep_pair_dict=keep_pair_dict)
    merge_data.to_csv(os.path.join(data_path, "submit", "fuxian_add_drop_dundoukong_substring_1.csv"),
                      columns=['id', 'negative', 'key_entity'], index=False, )

