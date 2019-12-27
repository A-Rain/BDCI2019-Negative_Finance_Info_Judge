import csv
import re
import os
from typing import List
from tqdm import tqdm
import logging
import pandas as pd
import torch
import copy
import numpy as np
import torch


raw_data_path = './'


# 合并相似的title 和 text
def concat(title, content):
    if type(title) is float or str(title) in str(content):
        return str(content)
    else:
        return str(title) + " " + str(content)


def get_jaccard_similar(title, text):
    if type(title) is float:
        return 0
    if type(text) is float:
        text = str(text)
    char_list_1 = []
    char_list_2 = []
    for char in title:
        char_list_1.append(char)
    for char in text:
        char_list_2.append(char)
    char_list_both = [char for char in char_list_1 if char in char_list_2]

    return len(char_list_both) / (len(char_list_1) + len(char_list_2) - len(char_list_both))
    # return len(char_list_1) + 0.0001 / len(char_list_both) + 0.0001
    # return len(char_list_both) / (len(char_list_1) if len(char_list_1) < len(char_list_2) else len(char_list_2))


def process_title(title, text, similar):
    if title == text or similar > 0.75:
        return ""
    else:
        return title


def cos_similar(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return round(dot_product / ((normA ** 0.5) * (normB ** 0.5)), 2)


train_data = pd.read_csv(os.path.join(raw_data_path,'Round2_train.csv'), encoding='utf-8')
test_data = pd.read_csv(os.path.join(raw_data_path,'round2_test.csv'), encoding='utf-8')

train_data["similar"] = train_data.apply(lambda x: get_jaccard_similar(x["title"], x["text"]), axis=1)
test_data["similar"] = test_data.apply(lambda x: get_jaccard_similar(x["title"], x["text"]), axis=1)

train_data["title"] = train_data.apply(lambda x: process_title(x["title"], x["text"], x["similar"]), axis=1)
# train_data["text"] = train_data.apply(lambda x: process_text(x["title"], x["text"], x["similar"]), axis=1)

test_data["title"] = test_data.apply(lambda x: process_title(x["title"], x["text"], x["similar"]), axis=1)
# test_data["text"] = test_data.apply(lambda x: process_text(x["title"], x["text"], x["similar"]), axis=1)

del train_data["similar"]
del test_data["similar"]

train_data["text"] = train_data.apply(lambda x: concat(x['title'], x['text']), axis=1)
test_data["text"] = test_data.apply(lambda x: concat(x['title'], x['text']), axis=1)

train_data["title"] = ""
test_data["title"] = ""

train_data.to_csv(os.path.join(raw_data_path,'Train_Data_round2.csv'), encoding='utf-8', index=False)
test_data.to_csv(os.path.join(raw_data_path,'Test_Data_round2.csv'), encoding='utf-8', index=False)






# 更改路径
train_data = pd.read_csv(os.path.join(raw_data_path,'Train_Data_round2.csv')) 
test_data = pd.read_csv(os.path.join(raw_data_path,'Test_Data_round2.csv'))
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

def process_chaohua_and_jing(text: str, entity_list: List[str]) -> str:
    new_entity_list = []
    for i  in entity_list:
        temp = i.strip()
        temp = re.sub('\?*','',temp)
        new_entity_list.append(temp)
    entity_list = new_entity_list
    pattern = "#[^#]*#"
    final_text = []
    while True:
        res = re.search(pattern, text)
        if res is None:
            final_text.append(text)
            break
        chunk_with_ent = res.group()
        curr_begin_idx, curr_end_idx = res.span()
        text_part1 = text[0: curr_begin_idx]
        text_part2 = text[curr_end_idx:]

        flag = True
        for ent in entity_list:
            if ent in chunk_with_ent:
                flag = False
                break
        final_text.append(text_part1)
        final_text.append("" if flag else chunk_with_ent)
        text = text_part2

    #final_text = re.sub('#|\[超话\]', "", "".join(final_text))
    final_text = re.sub('\[超话\]', "", "".join(final_text))
    return final_text


def process_at(text: str, entity_list: List[str]) -> str:
    pattern = "@[^(@|：| |,|，|。)]*( |，|,|：|:|。)|(//|回复)@[^(@|:)]*:"
    final_text = []
    while True:
        res = re.search(pattern, text)
        if res is None:
            final_text.append(text)
            break
        chunk_with_ent = res.group()
        curr_begin_idx, curr_end_idx = res.span()
        text_part1 = text[0: curr_begin_idx]
        text_part2 = text[curr_end_idx:]

        flag = True
        for ent in entity_list:
            if ent in chunk_with_ent:
                flag = False
                break

        final_text.append(text_part1)
        final_text.append("" if flag else chunk_with_ent)
        text = text_part2
    final_text = re.sub('@|//', "", "".join(final_text))
    return final_text


def process_laiyuan(text: str, entity_list: List[str]) -> str:
    pattern = "(文章|本文)?来源[：\:]\s?[^(，|。|\?|\||：|/||\s)]*[，。\?：/\|\s]"
    final_text = []
    while True:
        res = re.search(pattern, text)
        if res is None:
            final_text.append(text)
            break
        chunk_with_ent = res.group()
        curr_begin_idx, curr_end_idx = res.span()
        text_part1 = text[0: curr_begin_idx]
        text_part2 = text[curr_end_idx:]

        flag = True
        for ent in entity_list:
            if ent in chunk_with_ent:
                flag = False
                break

        final_text.append(text_part1)
        final_text.append("" if flag else chunk_with_ent)
        text = text_part2

    return "".join(final_text)



def eliminate_special_str(text: str, entity_list: List[str]) -> str:
    # 去除特殊字符
    regex1 = "①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩"
    text = re.sub(regex1, "", text)
    regex1 = "▽|￼▼|▲|■|█|▎|▌|★|►|▶|▼|—|\*|↑|→|°|·|ˇ|¤|é|§|●|…|☆|─|↓|�|✬|◆|▍|△|√|〖|〗|⊙|■|◆"
    text = re.sub(regex1, " ", text)
    # 去除微博@的数据
    # text = process_at(text, entity_list)
    # 处理##与超话
    text = re.sub('\[超话\]', " ", text)
    text = process_chaohua_and_jing(text, entity_list)
    # 去除js代码
    regex2 = "[a-z0-9'():/?\"=_\->]*[a-z'():/?\"=_\->][.:;=\"]+[a-z'():/?\"=_\->][a-z*0-9*'():/._?\"=>;\"]*"
    text = re.sub(regex2, " ", text)
    # 去掉js function的代码
    regex2 = "\(function\(\)\{.*\}?\(?\)?"
    text = re.sub(regex2, " ", text)
    # 去掉html标签
    regex2 = "<[^>]*>"
    text = re.sub(regex2, " ", text)
    # 去掉url
    regex2 = "(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?"
    text = re.sub(regex2, " ", text)
    # 去除 IMG
    regex3 = "\{IMG[^\}]*\}"
    text = re.sub(regex3, " ", text)

    # 去除记者 文章来源 责任编辑
    regex4 = "（[^（^）]*(记者|微信号|作者|通讯员)：?[^）^（]*）|【?编者按】?|新华社成都6月24日电"
    text = re.sub(regex4, " ", text)
    # 去除ID  以及编号
    regex5 = "（ID[^）]*）|\[[a-zA-Z0-9]*\]"
    text = re.sub(regex5, " ", text)
    # 去除 [doge] [cp] 等
    regex5 = "\[[a-zA-Z0-9]*\]"
    text5 = re.sub(regex5, " ", text)
    # 去除 点击上方蓝字
    regex6 = "【字号[^】]*】|点击上方蓝字|点击上方|【图】|回复使用道具举报|查看更多 |当前位置[：\:]|全文："
    text = re.sub(regex6, " ", text)
    regex6 = "收藏本站|收藏\([^\)]*\)|[0-9]*收藏|点击排行 |评论（[^）]*）|分享到腾讯微博|分享到新浪微博|赞[0-9]*|看全文 |(详细内容请)?点击："
    text = re.sub(regex6, " ", text)
    regex6 = "下载附件保存到相册|上传赞赏|支持([0-9]*人打赏)?分享|微信图片_[0-9]*|IMG_[0-9]*"
    text = re.sub(regex6, " ", text)
    regex6 = "资讯 |(网络)?电影 |电视剧 |综艺 |VIP(会员)? |首页 |导航 |娱乐 |片花 |脱口秀 |动漫 |游戏(视频|中心)? |搞笑 |微信 "
    text = re.sub(regex6, " ", text)
    regex6 = "体育 |教育 |儿童 |母婴 |生活 |健康 |军事 |汽车 |公益 |纪录片 |文学 |漫画 |热点 |风云榜 |全网影视 |应用商店 "
    text = re.sub(regex6, " ", text)
    regex6 = "大头 |爱奇艺号 |泡泡广场 |会员精选 |VR |泡泡 |旅游 |音乐 |时尚 |原创 |拍客 |科技 |奇秀直播 |直播中心 |商城 "
    text = re.sub(regex6, " ", text)
    regex6 = "网友评论 |最新评论 |暂无评论 |热度排行 |评论排行 |推荐 视频"
    text = re.sub(regex6, " ", text)
    regex7 = "\([^\(]*公众号：[^\)]*\)|(:\d*上传)?赞赏支持(\d*人)?(打赏)?分享:赞\d*\|收藏\(\d*\)"
    text = re.sub(regex7, " ", text)
    regex7 = "QQ空间 |微信 |朋友圈 |扫描二维码关注|[0-9]*(个回答|人关注)|主页 |企业动态 |正文 |舆情监测 |登录 |注册 |下载 |获取更多机会 |(顶|踩)一下 "
    text = re.sub(regex7, " ", text)
    regex7 = "\.(png|jpg)\s?\([^\)]*\)"
    text = re.sub(regex7, " ", text)

    text = process_laiyuan(text, entity_list)

    regex7 = "[\(（【\[][^(\(|（|【|\[)]*(编辑)：[^(\)|）)]*[\)）】\]]"
    text = re.sub(regex7, " ", text)

    regex7 = "(责任)?编辑[：\:]\s?[^(，|。|\?|\||：|/|\s)]*[，。\?：/\|\s]"
    text = re.sub(regex7, " ", text)


    # 去掉单独的数字,以及带括号的数字(124) （12435）
    regex_number = "[0-9]+ |[\(（\{][0-9a-zA-Z]+[\)）\}]"



    text = re.sub(regex_number, " ", text)
    # 去掉超长字符串
    regex_super_long = "[a-zA-Z0-9]{50,}"
    text = re.sub(regex_super_long, " ", text)


    # 去除股票代码
    regex8 = "[\(（][0-9]*[\.\-][a-zA-Z]*[\)）]|[\(（][a-zA-Z]*[:：][a-zA-Z]*[\)）]"
    text = re.sub(regex8, " ", text)
    # 去除&nbsp &quot
    regex9 = "&nbsp|&gt|&ldquo|&lsquo|&rsquo|&quot"
    text = re.sub(regex9, " ", text)
    # 去除 去除两个（时间）的情况
    regex11 = "\(?\d*/\d*/\d*\)?|\(?\d*-\d*-\d*\)?|\(?\d*\.\d*\.\d*\)?|\(?\d*年\d*月\d*日\)?|\(?\d*:\d*:\d*\)?"
    text = re.sub(regex11, " ", text)
    #去除 空格
    regex10 = " |" + "　" + "| " + "|	" + "|\s+|\s+" 
    text = re.sub(regex10, " ", text)
    return text


# 生成预处理后的文件
# 修改路径
for index, item in train_data.iterrows():
    text = item['text']
    entity_list = item['entity'].split(';')
    processed_text = eliminate_special_str(text,entity_list)
    train_data.iloc[index,2] = processed_text
train_data.to_csv(os.path.join(raw_data_path,'Train_Data_Title_processed_final.csv'),index=False)
for index, item in test_data.iterrows():
    text = item['text']
    entity_list = item['entity'].split(';')
    test_data.iloc[index,2] = processed_text
test_data.to_csv(os.path.join(raw_data_path,'Test_Data_Title_processed_final.csv'),index=False)


# 准备生成二分类的五折数据
train_data = pd.read_csv(os.path.join(raw_data_path,'Train_Data_Title_processed_final.csv')) 
test_data = pd.read_csv(os.path.join(raw_data_path,'Test_Data_Title_processed_final.csv'))
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
# 去除空entity
train_data = train_data[train_data['entity'].map(lambda x : len(x)>1)]
train_data.shape
test_data_no_entity = test_data[test_data['entity'].map(lambda x : len(x)<=1)]
test_data = test_data[test_data['entity'].map(lambda x : len(x)>1)]
test_data.shape
test_data_no_entity.shape
def select_test(context,title,entity_list,key_entity_list):
    return entity_list
def select_by_key_entity(context,title,entity_list,key_entity_list):
    new_list = [i for i in entity_list]
    new_list = sorted(new_list,key= lambda x:len(x),reverse=True)
    final_list = []
    for i in new_list:
        flag = True
        for j in final_list:
            if i in j and i and i not in key_entity_list :
                flag = False
                break
        if flag:               
            final_list.append(i)
    return final_list


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
        select_entity_list = select_by_key_entity(item['text'],'',entity_list, key_entity_list)
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

# 生成五折数据
# 修改路径

num = torch.randperm(len(train_entity)).tolist()
pd_list = []
for i in list(range(5)):
    left = i* int(len(num)/5)
    right = left+int(len(num)/5)
    temp_pd = train_entity.iloc[num[left:right]]
    pd_list.append(temp_pd)
for i in pd_list:
    print (len(i))
for i in range(5):
    temp_dev_pd = pd_list[i]
    train_cv = copy.deepcopy(train_entity)
    train_cv = train_cv.append(temp_dev_pd,ignore_index=True).drop_duplicates(keep=False)
    print('--',temp_dev_pd.shape,'--',train_cv.shape)
    path = os.path.join(os.path.join(raw_data_path,'fusai_cv_data','cv_'+str(i)))
    if not os.path.exists(path):
        os.makedirs(path)
    dev = pd.DataFrame({'index':temp_dev_pd['id'],'question':temp_dev_pd['entity'],
                        'sentence':temp_dev_pd['text'],'label':temp_dev_pd['negative']})
    dev_id = pd.DataFrame({'sentence':temp_dev_pd['text'],'id':temp_dev_pd['id']})
    train = pd.DataFrame({'index':train_cv['id'],'question':train_cv['entity'],
                          'sentence':train_cv['text'],'label':train_cv['negative']})
    test = pd.DataFrame({'index':test_entity['id'],'question':test_entity['entity'],'sentence':test_entity['text']})
    dev.to_csv(os.path.join(path,'dev.tsv'),sep='\t',index=False)
    dev_id.to_csv(os.path.join(path,'dev_id.tsv'),sep='\t',index=False)
    train.to_csv(os.path.join(path,'train.tsv'),sep='\t',index=False)
    test.to_csv(os.path.join(path,'test.tsv'),sep='\t',index=False)

# change length from cv file 
def find_all(sub,s):
    index_list = []
    index = s.find(sub)
    while index != -1:
        index_list.append(index)
        index = s.find(sub,index+1)

    if len(index_list) > 0:
        return index_list
    else:
        return -1

def get_span(loc,text,length):
    left = loc-length
    if loc+length < len(text):
        right= loc+length
    else:
        right = len(text)
    return text[left:right]
for i in range(5):
    path= os.path.join(raw_data_path,'fusai_cv_data','cv_'+str(i))
    data=['train','dev','test']
    save_path = os.path.join(raw_data_path,'fusai_cv_data_max512')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path_cv =os.path.join(save_path,'cv_'+str(i))
    #save_path_cv =save_path+'cv_'+str(i)
    if not os.path.exists(save_path_cv):
        os.mkdir(save_path_cv)
    for j in data:
        data_path = os.path.join(path,j+'.tsv')
        #data_path = path+j+'.tsv'
        train=pd.read_csv(data_path,sep='\t')
        print(data_path)
        num = 0
        for index,item in train.iterrows():
            entity = item['question']
            text = item['sentence']
            if len(text) > 512:
                if pd.isna(entity):
                    continue
                loc = text.find(entity)
                if loc == -1:
                    train.iloc[index,2]=text[0:512]
                else:
#                     print(loc,entity,'\n',text,[m.start() for m in re.finditer(entity, text)])
                    last_loc = find_all(entity,text)[-1]
                    if last_loc<512:
                        train.iloc[index,2]=text[0:512]
                    else:
                        head = text[0:100]
                        start_list = find_all(entity,text)[0:5]
                        if len(start_list)==0:
                            continue
                        span_length = (512-100)//len(start_list)
                        content = [get_span(i,text,span_length)for i in start_list]
                        text_target = head+' '.join(content)
                        train.iloc[index,2]=text_target
#                         print(item['index'],'\t',entity,'\t',len(start_list))
                        num +=1
        train.to_csv(os.path.join(save_path,'cv_'+str(i),j+'.tsv'),sep='\t',index=False)
        #train.to_csv(save_path+'cv_'+str(i)+'/'+j+'.tsv',sep='\t',index=False)
        print(num)


# change length from cv file 
## add <>
nan_num =0
for i in range(5):
    path= os.path.join(raw_data_path,'fusai_cv_data_max512','cv_'+str(i))
    data=['test','train','dev']
    save_path = os.path.join(raw_data_path,'fusai_cv_data_max512_span_fc')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path_cv =os.path.join(save_path,'cv_'+str(i))
    #save_path_cv =save_path+'cv_'+str(i)
    if not os.path.exists(save_path_cv):
        os.mkdir(save_path_cv)
    for j in data:
        data_path = os.path.join(path,j+'.tsv')
        #data_path = path+j+'.tsv'
        train=pd.read_csv(data_path,sep='\t')
        print(data_path)
        num = 0
        for index,item in train.iterrows():
            entity = item['question']
            text = item['sentence']
            if j != 'test':
                entity_all = train_data[train_data['id']==item['index']]['entity'].item().split(';')
            else:
                entity_all = test_data[test_data['id']==item['index']]['entity'].item().split(';')
            long = None
            try:
                for x in entity_all:
                    if type(x) is not float:
                        if  entity in x and len(x) != len(entity):
                            long = x
                            break
            except :
                long = None
                print('nan  entity: ',entity)
                continue
            if long is None:
                
                if type(entity) is not float and len(entity)>1 and '?'not in entity and '(' not in entity  and '（' not in entity and '*'not in entity:
                    print(entity)
                    text = re.sub(entity,'['+entity+']',text)
                    train.iloc[index,2]=text
                    num +=1
            else:
                if type(entity) is not float and len(entity)>1 and '?'not in entity and '(' not in entity  and '（' not in entity and '*'not in entity:
                    if type(long) is not float and len(long)>1 and '?'not in long and '(' not in long  and '（' not in long and '*'not in long:
                        print(entity)
                        text = re.sub(long,'@@@@@@@',text)
                        text = re.sub(entity,'['+entity+']',text)
                        text = re.sub('@@@@@@@',long,text)
                        train.iloc[index,2]=text
                        num +=1
        train.to_csv(os.path.join(save_path,'cv_'+str(i),j+'.tsv'),sep='\t',index=False)
        #train.to_csv(save_path+'cv_'+str(i)+'/'+j+'.tsv',sep='\t',index=False)
        print(num)