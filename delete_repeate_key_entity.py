import pandas as pd
import re
import numpy as np
from utils import data_path
import os


def is_same_Chinese(first_word, second_word):
    min_len = min(len(first_word), len(second_word))

    first_word_chinese = sorted([char for char in first_word if '\u4e00' <= char <= '\u9fff'])
    second_word_chinese = sorted([char for char in second_word if '\u4e00' <= char <= '\u9fff'])

    # 如果两个实体的中文部分相同，且各自至少包含一个中文字符，则返回true
    if first_word_chinese == second_word_chinese and len(first_word_chinese) > 0 and len(second_word_chinese) > 0:
        return True
    else:
        return False


def post_process(submit_data):
    id_add = []
    id_min = []
    for index, item in submit_data.iterrows():
        entity = item['entity'].split(';')
        for i in entity:
            if 'P2P' in i and len(i) < 6 and item['negative'] == 1 and i in item['key_entity']:
                id_min.append([i, item['id']])
            if '*' not in i and '(' not in i and ')' not in i and '?' not in i and len(i) > 1:
                result_add = re.search(i + '[^(。|，)]{1,10}(被裁定|被强制执行|被曝|自担保)', item['text'])
                result_min = re.search(i + '[^(。|，)]{1,10}(信誉十分好|逾期降低|符合国家规定|逾期率.*低至)', item['text'])
                if result_min is not None and i in item['key_entity']:
                    id_min.append([i, item['id']])
                if result_add is not None:
                    if item['negative'] == 0:
                        flag = True
                        for add_index in range(len(id_add)):
                            add_str, add_id = id_add[add_index]
                            if item['id'] == add_id:
                                if i in add_str:
                                    flag = False
                                    continue
                                elif add_str in i:
                                    flag = False
                                    id_add[add_index][0] = i
                        if flag:
                            id_add.append([i, item['id']])

    for item in id_add:
        # print(item[1])
        neg_pos = submit_data.columns.tolist().index('negative')
        key_entity_pos = submit_data.columns.tolist().index('key_entity')
        submit_data.iloc[submit_data[submit_data['id'] == item[1]].index.item(), neg_pos] = 1
        submit_data.iloc[submit_data[submit_data['id'] == item[1]].index.item(), key_entity_pos] = item[0]

    for item in id_min:
        # print(item[1])
        neg_pos = submit_data.columns.tolist().index('negative')
        key_entity_pos = submit_data.columns.tolist().index('key_entity')
        key_entity = submit_data.iloc[submit_data[submit_data['id'] == item[1]].index.item(), key_entity_pos].split(';')
        key_entity.remove(item[0])
        if len(key_entity) == 0:
            submit_data.iloc[submit_data[submit_data['id'] == item[1]].index.item(), neg_pos] = 0
            submit_data.iloc[submit_data[submit_data['id'] == item[1]].index.item(), key_entity_pos] = np.nan
        else:
            submit_data.iloc[submit_data[submit_data['id'] == item[1]].index.item(), key_entity_pos] = ';'.join(key_entity)

    return submit_data


if __name__ == '__main__':
    train_path = os.path.join(data_path, "preprocess", "Train_Data_round2.csv")
    test_path = os.path.join(data_path, "preprocess", "Test_Data_round2.csv")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    model_predict_result = pd.read_csv(os.path.join(data_path, "submit", "fuxian_replace_post.csv"))
    merge_data = test_data.merge(model_predict_result, left_on='id', right_on='id')

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

        final_key_entity_list = key_entity_list.copy()

        # 两重for循环，相当于形成(entity_outter, entity_inner) 实体对
        for i, key_entity_outter in enumerate(key_entity_list):
            for j, key_entity_inner in enumerate(key_entity_list):
                if i == j:
                    continue

                # 如果两个实体只是相差英文大小写，则将不在文本中的实体删掉
                if key_entity_outter.lower() == key_entity_inner.lower():
                    if key_entity_outter in cur_row['text'] and key_entity_inner not in cur_row['text']:
                        final_key_entity_list.remove(key_entity_inner)

                # # 如果两个entity的汉字部分一样
                if is_same_Chinese(key_entity_inner, key_entity_outter):
                    # 如果其中一个包含?或者空格，则将另一个实体删掉
                    if '?' in key_entity_inner or ' ' in key_entity_inner:
                        final_key_entity_list.remove(key_entity_outter)

        for i, entity_outter in enumerate(entity_list):
            for j, entity_inner in enumerate(entity_list):
                if i == j:
                    continue
                if is_same_Chinese(entity_inner, entity_outter):
                    if '?' in entity_inner and entity_outter in final_key_entity_list:
                        final_key_entity_list.remove(entity_outter)
                        final_key_entity_list.append(entity_inner)

        merge_data.loc[index:index, 'key_entity'] = ';'.join(final_key_entity_list)

    submit_data = post_process(merge_data)
    # cos_update(submit_data)
    submit_data.to_csv(os.path.join(data_path, "submit", "fuxian_replace_post_v2.csv"),
                       columns=['id', 'negative', 'key_entity'], index=False)
