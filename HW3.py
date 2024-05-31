import spacy
import random
from sklearn.metrics import classification_report

# 載入預訓練的英文模型
nlp = spacy.load("en_core_web_md")  # 使用帶有詞向量的英文模型

# 添加 NER 管道
if "ner" not in nlp.pipe_names:
    ner = nlp.create_pipe("ner")
    nlp.add_pipe('ner', last=True)
else:
    ner = nlp.get_pipe("ner")

# 讀取 CoNLL 格式數據
def read_conll(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = []
        tokens = []
        ner_tags = []
        for line in f:
            if line.strip() == "":
                if tokens:
                    sentences.append((tokens, ner_tags))
                    tokens = []
                    ner_tags = []
            else:
                parts = line.strip().split()
                if len(parts) == 2:
                    token, tag = parts
                    tokens.append(token)
                    ner_tags.append(tag)
                else:
                    print("Skipping invalid line:", line)
        if tokens:
            sentences.append((tokens, ner_tags))
    return sentences

# 讀取訓練數據
train_data_a = read_conll("entity-recognition-datasets/data/BTC/CONLL-format/data/a.conll")
train_data_b = read_conll("entity-recognition-datasets/data/BTC/CONLL-format/data/b.conll")
train_data_e = read_conll("entity-recognition-datasets/data/BTC/CONLL-format/data/e.conll")

# 將訓練數據合併到一個列表中
train_data = train_data_a #+ train_data_b + train_data_e

# 讀取測試數據
test_data_f = read_conll("entity-recognition-datasets/data/BTC/CONLL-format/data/f.conll")
#test_data_g = read_conll("entity-recognition-datasets/data/BTC/CONLL-format/data/g.conll")
#test_data_h = read_conll("entity-recognition-datasets/data/BTC/CONLL-format/data/h.conll")

# 將測試數據合併到一個列表中
test_data = train_data_a + test_data_f #+ test_data_g + test_data_h

# 添加標籤到 NER
added_labels = set()
for _, annotations in train_data:
    for ent in annotations:
        if ent not in added_labels:
            ner.add_label(ent)
            added_labels.add(ent)

# 開始訓練
optimizer = nlp.create_optimizer()
for itn in range(10):
    random.shuffle(train_data)
    losses = {}
    for texts, annotations in train_data:
        doc = nlp.make_doc(" ".join(texts))
        example = spacy.training.example.Example.from_dict(doc, {"entities": annotations})
        nlp.update([example], drop=0.5, losses=losses)
    print(losses)

# 評估模型
true_labels = []
pred_labels = []
for texts, annotations in test_data:
    doc = nlp(" ".join(texts))
    if doc.ents:  # 檢查是否有預測的命名實體
        pred_labels.extend([ent.label_ for ent in doc.ents])
    else:  # 如果沒有預測的命名實體，則將一個預設值添加到 pred_labels 中
        pred_labels.append("O")  # 這裡使用 "O" 作為預設值，你可以根據需要修改
    true_labels.extend(annotations)

# 打印預測標籤的數量
print("Length of predicted labels:", len(pred_labels))

# 計算評估指標並打印結果
print(classification_report(true_labels, pred_labels))
