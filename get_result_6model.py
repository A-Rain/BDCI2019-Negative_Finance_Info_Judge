import csv
import re
import os
from typing import List
from tqdm import tqdm
#import pkuseg
import logging
import pandas as pd

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
		

import os
import pandas as pd
import numpy as np
def get_pd(path_list):
    all_pd = []
    for path in path_list:
        for i in range(5):
            result_path = os.path.join(path,'cv_'+str(i),'result.csv')
            temp = pd.read_csv(result_path)
            all_pd.append(temp)
    return all_pd

def get_prob(path_list):
    all_prob = []
    for path in path_list:
        for i in range(5):
            result_path = os.path.join(path,'cv_'+str(i),'test_prob.npy')
            with open(result_path,'rb')as f:
                temp = np.load(f)
            all_prob.append(temp)
    return all_prob

	
	
data_prob = get_prob([
    '/userhome/project/pytorch-transformers-small/proc_data/test_all/bert_ext_l4',
    '/userhome/project/pytorch-transformers-small/proc_data/test_all/bert_ext_l2_pretrain',
    '/userhome/project/pytorch-transformers-master/proc_data/test_all/bert_large_4v',
    '/userhome/project/pytorch-transformers-master/proc_data/test_all/bert_large_4v_lr3',
    '/userhome/project/pytorch-transformers-master/proc_data/test_all/bert_large_4v_lr3_pretrain',
    '/userhome/project/pytorch-transformers-master/proc_data/test_all_span/bert_large_4v_lr2'
    
                 ])
				 


def softmax(x):
    """ softmax function """
    
    # assert(len(x.shape) > 1, "dimension must be larger than 1")
    # print(np.max(x, axis = 1, keepdims = True)) # axis = 1, 行
    
    x -= np.max(x, axis = 1, keepdims = True) #为了稳定地计算softmax概率， 一般会减掉最大的那个元素
    
    x = np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)
    
    return x

total_prob = 0
weight = [0.15 if i <25 else 0.25 for i in range(30)]
for index, i in enumerate(data_prob):
    print(index)
    total_prob += softmax(i)#*weight[index]


total_prob_result = total_prob.argmax(axis=1)

test_result_entity = pd.DataFrame({'id':data_pd[0]['id'],'negative':total_prob_result,'key_entity':test_entity['entity']})
test_result_entity['negative'].sum()

result_final = pd.DataFrame(columns=['id','negative','key_entity'])
neg_no_entity = pd.DataFrame(columns=['id','negative','key_entity'])
entity_filter_list = []
num = 0
total = 0
for index, item in test_result_entity.groupby('id'):
    total +=1
    negative = item['negative'].sum()
    if negative == 0:
        num = num +1
        key_entity = ''
        for i in item['key_entity']:
            entity = i         
            if entity in entity_filter_list  :
                if len(key_entity)==0:
                    key_entity = key_entity+entity
                else:
                    key_entity = key_entity+';'+entity
        if len(key_entity) ==0:
            s = pd.Series({'id':index,'negative':0,'key_entity':np.nan})
            neg_no_entity = neg_no_entity.append(s,ignore_index=True)
        else:
            s = pd.Series({'id':index,'negative':1,'key_entity':key_entity})
        result_final = result_final.append(s,ignore_index=True)
    else:

        key_entity = ''
        for i,it in item.iterrows():
            if it['negative'] == 1 :
                entity = it['key_entity']
                               
                if len(key_entity)==0:
                    key_entity = key_entity+entity
                else:
                    key_entity = key_entity+';'+entity
                    
        #entity_list = select(re.sub('#','',''.join(tokenizer.convert_ids_to_tokens(it['text']))),' ',key_entity.split(';'))
        #key_entity = ';'.join(entity_list)

        if len(key_entity) > 0:
            s = pd.Series({'id':index,'negative':1,'key_entity':key_entity})
            result_final = result_final.append(s,ignore_index=True)
        else:
            s = pd.Series({'id':index,'negative':0,'key_entity':np.nan})
            result_final = result_final.append(s,ignore_index=True)
            
print(num) 
test_o = pd.read_csv('/userhome/project/data_final/Test_Data_Title_processed.csv')
for index,item in test_o.iterrows():
    if item['id'] not in result_final['id'].tolist():
        result_final = result_final.append(pd.Series({'id':item['id'],'negative':1,'key_entity':item['entity']}),ignore_index=True)
        print(item)

result_final['id'] = result_final['id'].map(lambda x: int(x))
result_final['negative'] = result_final['negative'].map(lambda x: int(x))
result_final.to_csv('/userhome/project/result/test_all_tomerge_single_bert_large_4v_lr2_span.csv',index= False)