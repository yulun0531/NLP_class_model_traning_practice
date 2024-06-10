import spacy
from spacy.training import Example
from sklearn.metrics import classification_report
from spacy.tokens import DocBin
import random
def read_conll(file_path):
    """读取并处理CONLL格式的数据"""
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
            parts = line.strip().split()
            if len(parts) == 2:
                word, tag = parts
                sentence.append((word, tag))
            else:
                print(f"Skipping invalid line: {line.strip()}")
    if sentence:
        data.append(sentence)
    
    return data

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
        max_len = max(len(true_iob), len(pred_iob))
        true_iob += ['O'] * (max_len - len(true_iob))
        pred_iob += ['O'] * (max_len - len(pred_iob))
        y_true.extend(true_iob)
        y_pred.extend(pred_iob)
    # 打印分类报告
    print(classification_report(y_true, y_pred))

def convert_iob_to_spacy_format(data):
    """將IOB轉換成Spacy"""
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

def main():
    nlp = spacy.load("./model")

    # 读取并转换测试数据
    test_data_f = read_conll("entity-recognition-datasets/data/BTC/CONLL-format/data/a.conll")
    train_data_g = read_conll("entity-recognition-datasets/data/BTC/CONLL-format/data/b.conll")
    spacy_test_data = convert_iob_to_spacy_format(test_data_f+train_data_g)
    
    # 评估模型
    evaluate_model(nlp, spacy_test_data)
if __name__ == "__main__":
    main()
