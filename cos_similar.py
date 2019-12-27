import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
from utils import concat, data_path
import os


train_path = os.path.join(data_path, "preprocess", "Train_Data_round2.csv")
test_path = os.path.join(data_path, "preprocess", "Test_Data_round2.csv")


def jaccard_similar(text1, text2):
    char_list_1 = set([char for char in text1 if char not in [" "]])
    char_list_2 = set([char for char in text2 if char not in [" "]])
    char_list_both = set([char for char in char_list_1 if char in char_list_2])

    return len(char_list_both) / (len(char_list_1) + len(char_list_2) - len(char_list_both))


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


def get_most_jaccard_similar(idx, data):
    most_jaccard_similar = 0
    max_idx = 0
    test_text = data.iloc[idx]["text"]
    for i in range(len(data)):
        if i == idx:
            continue
        if (i > idx - 30) and (i < idx + 30) and data.iloc[i]["category"] == "Train":
            score = jaccard_similar(test_text, data.iloc[i]["text"])
            if score > most_jaccard_similar:
                most_jaccard_similar = score
                max_idx = i
    return most_jaccard_similar, max_idx


def get_most_similar(idx, data, vector_list):
    max_cos_similar = 0
    max_idx = 0
    vector = vector_list[idx]
    for i in range(len(vector_list)):
        if i == idx:
            continue
        if (i > idx - 30) and (i < idx + 30) and data.iloc[i]["category"] == "Train":
            cos = cos_similar(vector, vector_list[i])
            if cos > max_cos_similar:
                max_cos_similar = cos
                max_idx = i
    return max_cos_similar, max_idx


def get_data_cluster(train_data, test_data, submit_file, sorted_by):
    train_data["text"] = train_data.apply(lambda x: concat(x["title"], x["text"]), axis=1)
    test_data["text"] = test_data.apply(lambda x: concat(x["title"], x["text"]), axis=1)
    train_data["category"] = "Train"
    test_data["category"] = "Test"

    if os.path.exists(os.path.join(data_path, "submit", submit_file)):
        submit = pd.read_csv(os.path.join(data_path, "submit", submit_file), encoding='utf-8')
        test_data = pd.merge(test_data, submit, on='id', how='left')
    else:
        test_data["negative"] = 0
        test_data["key_entity"] = ""

    data = pd.concat((train_data[["id", "text", "category", "negative", "entity", "key_entity"]],
                      test_data[["id", "text", "category", "negative", "entity", "key_entity"]]),
                     axis=0).reset_index(drop=True)

    data.sort_values(by=sorted_by, inplace=True)
    data.to_csv(os.path.join(data_path, "Data_Cluster.csv"), encoding='utf-8', index=False)


def get_train_test_cos_similar(cos_file):
    data = pd.read_csv(os.path.join(data_path, "Data_Cluster.csv"), encoding='utf-8')
    data = data[["id", "text", "category", "negative", "entity", "key_entity"]]
    data["text"] = data["text"].apply(lambda x: " ".join([word for word in jieba.cut(str(x)) if x not in [" "]]))
    counter = TfidfVectorizer(use_idf=True, max_features=10000, ngram_range=(1, 3))
    counter.fit(data['text'])
    weight = counter.transform(data['text']).toarray()
    test_max_cos_similar = []
    test_max_cos_text = []
    test_max_cos_id = []
    test_max_negative = []
    test_max_entity = []
    test_max_key_entity = []
    for i in range(len(data)):
        print(i)
        if data.iloc[i]["category"] == "Train":
            test_max_cos_similar.append("")
            test_max_cos_id.append("")
            test_max_cos_text.append("")
            test_max_negative.append("")
            test_max_entity.append("")
            test_max_key_entity.append("")
        else:
            max_cos_similar, max_idx = get_most_similar(i, data, weight)
            test_max_cos_id.append(data.iloc[max_idx]["id"])
            test_max_cos_similar.append(max_cos_similar)
            test_max_cos_text.append(data.iloc[max_idx]["text"].replace(" ", ""))
            test_max_negative.append(data.iloc[max_idx]["negative"])
            test_max_entity.append(data.iloc[max_idx]["entity"])
            test_max_key_entity.append(data.iloc[max_idx]["key_entity"])
    data["cos_similar"] = test_max_cos_similar
    data["similar_id"] = test_max_cos_id
    data["similar_text"] = test_max_cos_text
    data["text"] = data["text"].apply(lambda x: x.replace(" ", ""))
    data['similar_negative'] = test_max_negative
    data['similar_entity'] = test_max_entity
    data['similar_key_entity'] = test_max_key_entity

    data[["id", "text", "similar_text", "cos_similar", "similar_id", "category",
          "entity", "negative", "key_entity", "similar_negative", "similar_entity", "similar_key_entity"]]\
        .to_csv(os.path.join(data_path, cos_file), encoding='utf-8', index=False)


def get_train_test_jaccard_similar():
    data = pd.read_csv(os.path.join(data_path, "Data_Cluster.csv"), encoding='utf-8')
    data = data[["id", "text", "category", "negative", "entity", "key_entity"]]
    test_max_jaccard_similar = []
    test_max_jaccard_text = []
    test_max_jaccard_id = []
    test_max_negative = []
    test_max_entity = []
    test_max_key_entity = []
    for i in range(len(data)):
        print(i)
        if data.iloc[i]["category"] == "Train":
            test_max_jaccard_similar.append("")
            test_max_jaccard_id.append("")
            test_max_jaccard_text.append("")
            test_max_negative.append("")
            test_max_entity.append("")
            test_max_key_entity.append("")
            continue
        max_jaccard_similar, max_idx = get_most_jaccard_similar(i, data)
        test_max_jaccard_id.append(data.iloc[max_idx]["id"])
        test_max_jaccard_similar.append(max_jaccard_similar)
        test_max_jaccard_text.append(data.iloc[max_idx]["text"].replace(" ", ""))
        test_max_negative.append(data.iloc[max_idx]["negative"])
        test_max_entity.append(data.iloc[max_idx]["entity"])
        test_max_key_entity.append(data.iloc[max_idx]["key_entity"])
    data["jaccard_similar"] = test_max_jaccard_similar
    data["similar_id"] = test_max_jaccard_id
    data["similar_text"] = test_max_jaccard_text
    data["text"] = data["text"].apply(lambda x: x.replace(" ", ""))
    data['similar_negative'] = test_max_negative
    data['similar_entity'] = test_max_entity
    data['similar_key_entity'] = test_max_key_entity

    data[["id", "text", "similar_text", "jaccard_similar", "similar_id", "category",
          "entity", "negative", "key_entity", "similar_negative", "similar_entity", "similar_key_entity"]]\
        .to_csv(os.path.join(data_path, "test_jaccard_text.csv"), encoding='utf-8', index=False)


if __name__ == "__main__":
    train_data = pd.read_csv(os.path.join(train_path), encoding='utf-8')
    test_data = pd.read_csv(os.path.join(test_path), encoding='utf-8')
    get_data_cluster(train_data, test_data,
                     "fuxian_result.csv", ["entity", "text"])
    get_train_test_cos_similar("test_cos_text_v2_fuxian.csv")

    get_data_cluster(train_data, test_data,
                     "fuxian_result.csv", ["text"])
    get_train_test_cos_similar("test_cos_text_fuxian.csv")
    # get_train_test_jaccard_similar()
