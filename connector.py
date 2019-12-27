import pandas as pd
import os
import re
from utils import data_path


# def get_comma_entiy(text, entity_list, comma_threshold=5, none_count=0):
#     if len(entity_list) < 5:
#         return None, None, None, none_count
#     try:
#         temp_num = comma_threshold
#
#         temp_temp = '|'.join(entity_list)
#         regex = '('
#         for i in entity_list:
#             regex += '(' + i + '、' + ')|'
#         regex = regex[:-1] + '|(^(' + temp_temp + ').{1,10}、)' + ')' + '{' + str(temp_num) + ',}'
#
#         x = re.search(regex, text)
#         if x is not None:
#             start, end = x.span()
#             if len(x.group().split('、')) < 5:
#                 print(text)
#             return x.group(), start, end, none_count
#         else:
#             return None, None, None, none_count
#     except :
#         none_count += 1
#         return None, None, None, none_count


def get_comma_entity_list(text, entity_list, comma_threshold=2, none_count=0, all_entity=None, split_character='、'):
    entity_max_length = 0
    # 两种策略，第一种是所有entity中的最大长度
    # for entity in all_entity:
    #     if len(entity) > entity_max_length:
    #         entity_max_length = len(entity)
    # 第二种策略，entity_list中实体的最大长度
    for entity in entity_list:
        if len(entity) > entity_max_length:
            entity_max_length = len(entity)

    result_bound = []
    result_match_entity_list = []
    left = -1
    right = -1
    # comma_list = text.split(split_character)
    comma_list = re.split(split_character, text)

    if len(comma_list) <= 1:
        return None, None, None, none_count

    for i, item in enumerate(comma_list):
        if item in all_entity:
            if left == -1:
                left = i
            right = i
        else:
            if right != -1:
                result_bound.append([left, right])
                result_match_entity_list.append(comma_list[left: right+1])
                left = -1
                right = -1
                break

    # 对应第一种策略
    # for i, pair in enumerate(result_bound):
    #     left = pair[0]
    #     right = pair[1]
    #     if left != 0:
    #         flag = False
    #         # 如果这个最长的词正好是个实体
    #         if comma_list[left - 1][-entity_max_length:] in all_entity:
    #             flag = True
    #             result_match_entity_list[i].insert(0, comma_list[left - 1][-entity_max_length:])
    #         # 如果某个实体在这个串里
    #         for enti in all_entity:
    #             if enti in comma_list[left - 1][-entity_max_length:]:
    #                 flag = True
    #                 result_match_entity_list[i].insert(0, enti)
    #                 break
    #         if flag is True:
    #             left = left - 1
    #
    #     if right != len(comma_list) - 1:
    #         flag = False
    #         # 如果这个最长的词正好是个实体
    #         if comma_list[right + 1][:entity_max_length] in all_entity:
    #             flag = True
    #             result_match_entity_list[i].append(comma_list[right + 1][:entity_max_length])
    #         # 如果某个实体在这个串里
    #         for enti in all_entity:
    #             if enti in comma_list[right + 1][:entity_max_length]:
    #                 flag = True
    #                 result_match_entity_list[i].append(enti)
    #                 break
    #         if flag is True:
    #             right = right + 1

    # 对应第二种策略
    for i, pair in enumerate(result_bound):
        left = pair[0]
        right = pair[1]
        if left != 0:
            flag = False
            # 如果这个最长的词正好是个实体
            if comma_list[left - 1][-entity_max_length:] in entity_list:
                flag = True
                result_match_entity_list[i].insert(0, comma_list[left - 1][-entity_max_length:])
            # 如果某个实体在这个串里， 且应该匹配最长的，比如e租宝和租宝时，就应该加入e租宝
            if flag is False:
                max_match_entity_len = -1
                max_match_e = None
                for enti in entity_list:
                    if enti in comma_list[left - 1][-entity_max_length:]:
                        flag = True
                        if len(enti) > max_match_entity_len:
                            max_match_entity_len = len(enti)
                            max_match_e = enti
                if max_match_e != None:
                    result_match_entity_list[i].insert(0, max_match_e)

            if flag is True:
                left = left - 1

        if right != len(comma_list) - 1:
            flag = False
            # 如果这个最长的词正好是个实体
            if comma_list[right + 1][:entity_max_length] in entity_list:
                flag = True
                result_match_entity_list[i].append(comma_list[right + 1][:entity_max_length])
            if flag is False:
                # 如果某个实体在这个串里, 且应该匹配最长的，比如e租宝和租宝时，就应该加入e租宝
                max_match_entity_len = -1
                max_match_e = None
                for enti in entity_list:
                    if enti in comma_list[right + 1][:entity_max_length]:
                        flag = True
                        if len(enti) > max_match_entity_len:
                            max_match_entity_len = len(enti)
                            max_match_e = enti
                if max_match_e != None:
                    result_match_entity_list[i].append(max_match_e)
            if flag is True:
                right = right + 1

        result_bound[i] = [left, right]

    if len(result_bound) == 0:
        return None, None, None, none_count


    max_gap = -1
    max_gap_left = -1
    max_gap_right = -1
    max_i = -1
    for i, item in enumerate(result_bound):
        if item[1] - item[0] > max_gap:
            max_gap = item[1] - item[0]
            max_gap_left = item[0]
            max_gap_right = item[1]
            max_i = i
    if max_gap + 1 < comma_threshold:
        return None, None, None, none_count

    if len(result_bound) > 1:
        assert 0 == 1
        print('匹配到了{}段'.format(len(result_bound)))
    else:
        none_count += 1
        max_match_entity = comma_list[max_gap_left:max_gap_right + 1]
        max_match_entity_sub_no_ralate = result_match_entity_list[max_i]
        # print(result_bound)

    # 去掉不在‘entity’里的实体
    final_result = []

    for item in max_match_entity_sub_no_ralate:
        if item in entity_list and item.strip() != '':
            final_result.append(item)

    return final_result, None, None, none_count


test_data = pd.read_csv(os.path.join(data_path, "preprocess", "Test_Data_round2.csv"))
predict_result_data = pd.read_csv(os.path.join(data_path, "submit", "fuxian_add_drop.csv"))
merge_data = test_data.merge(predict_result_data, left_on='id', right_on='id')

all_entity_list = []
for index, cur_row in test_data.iterrows():
    entity_list = str(cur_row['entity']).split(';')
    for item in entity_list:
        if item.strip() == '':
            continue
    all_entity_list.extend(entity_list)
all_entity_set = set(all_entity_list)
all_entity_set.remove('')
all_entity_set.remove(' ')


def process_split_character(character, threshold):
    print('现在正在处理‘{}’符号'.format(character))
    none_count = 0
    count = 0
    id_list = []



    for index, cur_row in merge_data.iterrows():
        if cur_row['id'] == 14848:
            print('haha')

        entity_list = str(cur_row['entity']).split(';')

        # print(str(cur_row['text']).split('、'))
        match_entity_list, start, end, none_count = get_comma_entity_list(cur_row['text'], entity_list, comma_threshold=threshold,none_count=none_count, all_entity=all_entity_set, split_character=character)
        if match_entity_list is not None:
            # print(group)
            # match_entity_list = group.split('、')
            # if not pd.isna(cur_row['key_entity']):
            if int(cur_row['negative']) == 1:
                key_entity_list = str(cur_row['key_entity']).split(';')
                remove_empty_entity_list = []
                for key_e in key_entity_list:
                    if key_e.strip() != '':
                        remove_empty_entity_list.append(key_e)
                key_entity_list = remove_empty_entity_list
                origin_key_entity_list = key_entity_list.copy()

                # 如果存在交集，且。。。，且 需要加入的个数 小于等于 交集的个数
                if len(set(match_entity_list)&set(key_entity_list)) and len(set(match_entity_list)&set(key_entity_list)) != len(set(match_entity_list)) and (len(set(match_entity_list)&set(key_entity_list)) >= len(set(match_entity_list) - (set(match_entity_list)&set(key_entity_list)))):
                    count += 1
                    key_entity_list.extend(match_entity_list)
                    final_key_entity = []
                    for i in key_entity_list:
                        if len(i) > 0:
                            final_key_entity.append(i)
                    new_key_entity = ';'.join(list(set(final_key_entity)))
                    merge_data.loc[index:index, 'key_entity'] = new_key_entity
                    id_list.append(cur_row['id'])
                    print(cur_row['id'])
                    count_of_add = len(set(match_entity_list) - (set(match_entity_list)&set(origin_key_entity_list)))
                    count_of_add_v2 = len(new_key_entity.split(';')) - len(origin_key_entity_list)
                    assert count_of_add == count_of_add_v2
                    print('匹配到的为{}， 匹配的个数为{}'.format(match_entity_list, len(match_entity_list)))
                    print('加入的为{}, 加的个数为：{}'.format(str(set(match_entity_list) - (set(match_entity_list)&set(origin_key_entity_list))), count_of_add))
                    print('交集为{}, 交集数量为{}'.format(str(set(match_entity_list)&set(origin_key_entity_list)), len(set(match_entity_list)&set(origin_key_entity_list))))

                # 如果存在交集，且 需要加入的个数 大于 交集的个数
                if len(set(match_entity_list)&set(origin_key_entity_list)) and (len(set(match_entity_list)&set(origin_key_entity_list)) < len(set(match_entity_list) - (set(match_entity_list)&set(origin_key_entity_list)))):
                    count += 1
                    intersection = set(match_entity_list) & set(origin_key_entity_list)
                    # 模型预测的key_entity 减去交集
                    new_key_entity = ';'.join(list(set(origin_key_entity_list) - intersection))
                    merge_data.loc[index:index, 'key_entity'] = new_key_entity
                    id_list.append(cur_row['id'])
                    print(cur_row['id'])
                    count_of_sub = len(set(match_entity_list)&set(origin_key_entity_list))
                    if new_key_entity == '':
                        count_of_sub_v2 = len(origin_key_entity_list)
                    else:
                        count_of_sub_v2 = len(origin_key_entity_list) - len(new_key_entity.split(';'))
                    assert count_of_sub == count_of_sub_v2
                    print('匹配到的为{}, 匹配的个数为{}'.format(match_entity_list, len(match_entity_list)))
                    print('减去的为{}, 减的个数为：{}'.format((set(match_entity_list)&set(origin_key_entity_list)), count_of_sub))
                    print('交集为{}, 交集数量为{}'.format(str(set(match_entity_list)&set(origin_key_entity_list)), len(set(match_entity_list)&set(origin_key_entity_list))))

    # print('except count is{}'.format(group))
    print(count)
    # print(none_count)
    print(id_list)


process_split_character('、', threshold=3)
process_split_character('，', threshold=3)
process_split_character(',', threshold=3)
# process_split_character(';', threshold=3)
process_split_character(' ', threshold=3)
# process_split_character('和', threshold=2)
# process_split_character('[，]', threshold=3)
merge_data["negative"] = merge_data["key_entity"].apply(lambda x: 0 if type(x) is float or x == "" else 1)
merge_data.to_csv(os.path.join(data_path, "submit", "fuxian_add_drop_dundoukong.csv"),
                  columns=['id', 'negative', 'key_entity'], index=False)