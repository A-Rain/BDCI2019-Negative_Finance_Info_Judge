import csv
import re
import os
from typing import List
from tqdm import tqdm
#import pkuseg
import logging
import pandas as pd

# please change data_path
import pandas as pd
train_data = pd.read_csv('/userhome/project/data_final/Train_Data_Title_processed_anzhaochusai.csv') 
test_data = pd.read_csv('/userhome/project/data_final/Test_Data_Title_processed_anzhaochusai.csv')
test_entity_null_id = test_data[test_data['entity'].isnull()]['id']
test_data['entity'] = test_data['entity'].fillna(' ')
test_data['title'] = test_data['title'].fillna('')
test_data['text'] = test_data['text'].fillna('')
test_data['text']= test_data.apply(lambda x:x['title']+' '+x['text']if x['title']!=x['text'] else x['text'],axis=1)
test_data['text'] = test_data.apply(lambda x:x['text'].strip(),axis=1)
train_data['entity'] = train_data['entity'].fillna(' ')
train_data['title'] = train_data['title'].fillna('')
train_data['text'] = train_data.apply(lambda x: x['title']+' '+x['text'] if x['title'] != x['text'] else x['text'],axis=1)
train_data['text']  = train_data.apply(lambda x:x['text'].strip(),axis=1)

train_data.shape

good_entity = []
for index,item in train_data.iterrows():
    if item['negative']== 1:
        entity_list = item['key_entity'].split(';')
        
        for i in entity_list:
            
            if  i not in item['text']:
                if pd.notnull(item['title']) and i in item['title']:
                    continue
                else:
                    good_entity.append(i)

len(good_entity)

# 去除空entity
train_data = train_data[train_data['entity'].map(lambda x : len(x)>1)]
train_data.shape

test_data_no_entity = test_data[test_data['entity'].map(lambda x : len(x)<=1)]
test_data = test_data[test_data['entity'].map(lambda x : len(x)>1)]
test_data.shape

test_data_no_entity.shape

def select_train(context,title,entity_list,key_entity_list):
    new_list = [i for i in entity_list]
#     扔掉不出现在text中的实体
#     for i in entity_list:
#         if (i not in context) and (i not in title) and i not in train_total_entity:
#             new_list.remove(i)
#             print(entity_list,i)
    new_list = sorted(new_list,key= lambda x:len(x),reverse=True)
    final_list = []
    for i in new_list:
        flag = True
        for j in final_list:
            if i in j and i not in key_entity_list:
                flag = False
                break
        if flag:
            final_list.append(i)
    return final_list


def select_test(context,title,entity_list,key_entity_list):
    new_list = [i for i in entity_list]
#     扔掉不出现在text中的实体
#     for i in entity_list:
#         if (i not in context) and (i not in title) and i not in train_total_entity:
#             new_list.remove(i)
#             print(entity_list,i)
#     new_list = sorted(new_list,key= lambda x:len(x),reverse=True)
#     final_list = []
#     for i in new_list:
#         flag = True
#         for j in final_list:
#             if i in j:
#                 flag = False
#                 break
#         if flag:
#             final_list.append(i)
    return entity_list



train_entity = pd.DataFrame(columns=['id','text','entity','negative'])
test_entity = pd.DataFrame(columns=['id','text','entity'])


for index,item in train_data.iterrows():
    if item['negative'] == 0:
        entity_list = item['entity'].split(';')
        for i in entity_list:
            train_entity = train_entity.append(pd.Series({'id':item['id'],'text':item['text'],'entity':i,'negative':0}),ignore_index=True)
    else:
        entity_list = item['entity'].split(';')
        key_entity_list = item['key_entity'].split(';')
        select_entity_list = select_train(item['text'],'',entity_list,key_entity_list)
        for i in select_entity_list:
            if i not in key_entity_list:
                train_entity = train_entity.append(pd.Series({'id':item['id'],'text':item['text'],'entity':i,'negative':0}),ignore_index=True)
            else:
                train_entity = train_entity.append(pd.Series({'id':item['id'],'text':item['text'],'entity':i,'negative':1}),ignore_index=True)

for index,item in test_data.iterrows():
    entity_list = item['entity'].split(';')
    select_entity_list = select_test(item['text'],'',entity_list,key_entity_list)
    if len(select_entity_list) == 0:
        print(entity_list)
    for i in select_entity_list:
        test_entity = test_entity.append(pd.Series({'id':item['id'],'text':item['text'],'entity':i}),ignore_index=True)

import pandas as pd

small_path='/userhome/project/pytorch-transformers-small/proc_data/test_all_nocls'
large_path = '/userhome/project/pytorch-transformers-master/proc_data/test_all_nocls'
large_path_span = '/userhome/project/pytorch-transformers-master/proc_data/test_all_span_nocls'
base_list = ['bert_ext_l2','bert_ext_l2_pretrain','bert_large_4v_lr2','bert_large_4v_lr3','bert_large_4v_lr3_pretrain','bert_large_4v_lr3']
import numpy as np
all_cv=[]
for cv in range(5):
    cv_result=[]
    for j in range(len(base_list)):
        i = base_list[j]
        if j<=1 :
            print(i)
            path_temp  = small_path
        elif j<=4 :
            path_temp = large_path
        else:
            print('span')
            path_temp = large_path_span
        path = os.path.join(path_temp,i,'cv_'+str(cv),'test_prob.npy')
        with open(path,'rb')as f:
            temp = np.load(f)
            cv_result.append(temp)
    all_cv.append(cv_result)

def softmax(x):
    """ softmax function """
    
    # assert(len(x.shape) > 1, "dimension must be larger than 1")
    # print(np.max(x, axis = 1, keepdims = True)) # axis = 1, 行
    
    x -= np.max(x, axis = 1, keepdims = True) #为了稳定地计算softmax概率， 一般会减掉最大的那个元素
    
    x = np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)
    
    return x

softmax(all_cv[0][0])

import copy
pd_result = []
for i in range(5):
    test_entity_result = copy.deepcopy(test_entity)
    cv = all_cv[i]
    for j in range(len(base_list)):
        test_entity_result[base_list[j]]=softmax(cv[j])[:,1]
    pd_result.append(test_entity_result)

def select(entity_list):
    new_list = [i for i in entity_list]
#     不扔掉不出现在text中的实体
#     for i in entity_list:
#         if (i not in context) and (i not in title) and i not in good_entity:
#             new_list.remove(i)
    new_list = sorted(new_list,key= lambda x:len(x),reverse=True)
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

temp = pd.read_csv('/userhome/project/duibi/all_cv_0_result.csv')

temp['entity']=temp['entity'].fillna('')

temp_new = pd.DataFrame()
for index,item in temp.groupby('id'):
    entity_list = list(item['entity'])
    entity_list_new=select(entity_list)
    for index_t ,item_t in item.iterrows():
        if item_t['entity'] in entity_list_new:
            temp_new = temp_new.append(item_t)
