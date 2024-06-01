import spacy
from spacy.training import Example
from sklearn.metrics import classification_report
import random

def read_conll(file_path):
    """读取CONLL格式的数据并返回训练数据"""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    data = []
    sentence = []
    for line in lines:
        if line.strip() == '':
            if sentence:
                data.append(sentence)
                sentence = []
        else:
            word, tag = line.strip().split()
            sentence.append((word, tag))
    if sentence:
        data.append(sentence)
    
    return data

def convert_iob_to_spacy_format(data):
    """将IOB格式的数据转换为spaCy格式"""
    spacy_data = []
    for sentence in data:
        words = [token[0] for token in sentence]
        entities = []
        start = 0
        end = 0
        for word, tag in sentence:
            end = start + len(word)
            if tag != 'O':
                entity_label = tag.split('-')[1]
                if tag.startswith('B-'):
                    entities.append((start, end, entity_label))
            start = end + 1  # Adding 1 for the space between words
        spacy_data.append((words, {'entities': entities}))
    return spacy_data

def create_training_data(data):
    """创建训练数据"""
    nlp = spacy.blank('en')
    db = DocBin()
    for text, annotations in data:
        doc = nlp.make_doc(" ".join(text))
        example = Example.from_dict(doc, annotations)
        db.add(example.reference)
    return db

def train_model(train_data):
    """训练NER模型"""
    nlp = spacy.blank("en")
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    
    # 添加标签
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    
    # 配置训练参数
    nlp.begin_training()
    optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    
    # 开始训练
    with nlp.disable_pipes(*other_pipes):
        for iteration in range(10):  # 设置迭代次数为10
            random.shuffle(train_data)
            losses = {}
            for text, annotations in train_data:
                doc = nlp.make_doc(" ".join(text))
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.5, losses=losses, sgd=optimizer)
            print(f"Iteration {iteration}: {losses}")

    return nlp

def evaluate_model(nlp, test_data):
    """评估NER模型并生成classification_report"""
    y_true = []
    y_pred = []

    for text, annotations in test_data:
        doc = nlp(" ".join(text))
        true_labels = annotations['entities']
        pred_labels = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

        # 将标签转化为IOB格式
        true_iob = ['O'] * len(doc)
        for start, end, label in true_labels:
            true_iob[start:end] = ['B-' + label] + ['I-' + label] * (end - start - 1)
        
        pred_iob = ['O'] * len(doc)
        for start, end, label in pred_labels:
            pred_iob[start:end] = ['B-' + label] + ['I-' + label] * (end - start - 1)
        
        y_true.extend(true_iob)
        y_pred.extend(pred_iob)

    # 打印分类报告
    print(classification_report(y_true, y_pred))

def main():
    # 读取并转换训练数据
    train_data_a = read_conll("entity-recognition-datasets/data/BTC/CONLL-format/data/a.conll")
    spacy_train_data = convert_iob_to_spacy_format(train_data_a)
    
    # 训练模型
    nlp = train_model(spacy_train_data)
    
    # 读取并转换测试数据
    test_data_f = read_conll("entity-recognition-datasets/data/BTC/CONLL-format/data/f.conll")
    spacy_test_data = convert_iob_to_spacy_format(test_data_f)
    
    # 评估模型
    evaluate_model(nlp, spacy_test_data)

if __name__ == "__main__":
    main()
