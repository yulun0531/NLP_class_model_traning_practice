# 导入程序运行必需的库
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, optimizers
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import re
import os

#-------------------数据预处理----------------------

# 根据路径打开文件并提取每个邮件中的文本
def getMailText(mailPath):
    mail = open(mailPath, "r", encoding="gb2312", errors='ignore')
    mailTextList = [text for text in mail]
    # 去除邮件头
    XindexList = [mailTextList.index(i) for i in mailTextList if re.match("[a-zA-Z0-9]", i)]
    textBegin = max(XindexList) + 1
    text = str(''.join(mailTextList[textBegin:]))  # 将文本内容转换为字符串
    # 去空格分隔符及一些特殊字符
    text = re.sub('\s+', '', re.sub("\u3000", "", re.sub("\n", "", text)))
    return text

# 通过index文件获取所有文件路径及标签值
def getPaths_Labels():
    targets = open("./trec06p/full/index", "r", encoding="gb2312", errors='ignore')
    targetList = [t for t in targets]
    newTargetList = [target.split() for target in targetList if len(target.split()) == 2]  
    pathList = [path[1].replace('..', './trec06p') for path in newTargetList]
    label_list = [label[0] for label in newTargetList]
    return pathList, label_list

# 获取所有文本
def getAllText(pathList):
    content_list = [getMailText(filePath) for filePath in pathList]
    return content_list

# 0 为垃圾邮件 1 为正常邮件
def transform_label(label_list):
    list = [0 if label == "spam" else 1 for label in label_list]
    return list

#-------------------文本分类----------------------
class TextClassification():
    # 为超参数赋值
    def config(self):
        self.seq_length = 600    # 允许句子最大长度
        self.embedding_dim = 64  # 词向量维度
        self.hidden_dim = 32  # 全连接层神经元
        self.dropout_keep_prob = 0.5  # dropout保留比例
        self.learning_rate = 1e-3  # 学习率
        self.batch_size = 128   # 每批训练大小
        self.num_iteration = 5000 # 迭代次数
        self.print_per_batch = self.num_iteration // 100 # 每迭代5000/100=50次打印一次

    def __init__(self, content_list, label_list):
        self.config()
        train_X, test_X, train_y, test_y = train_test_split(content_list, label_list)
        self.train_content_list = train_X
        self.train_label_list = train_y
        self.test_content_list = test_X
        self.test_label_list = test_y
        self.content_list = self.train_content_list + self.test_content_list
        self.autoGetNumClasses()
        self.prepareData()

    def autoGetNumClasses(self):
        label_list = self.train_label_list + self.test_label_list
        self.num_classes = np.unique(label_list).shape[0]

    def prepareData(self):
        self.labelEncoder = LabelEncoder()
        self.labelEncoder.fit(self.train_label_list)

    # 文本标签列表label_list转换为预测目标值Y
    def label2Y(self, label_list):
        y = self.labelEncoder.transform(label_list)
        Y = keras.utils.to_categorical(y, self.num_classes)
        return Y

    # 搭建卷积神经网络模型
    def buildModel(self):
        vocab_size = 10000  # 设置词汇表大小，具体根据数据集的情况来定
        self.model = tf.keras.Sequential([
            layers.Input(shape=(self.seq_length,), dtype="string"),  # 输入层
            layers.Embedding(input_dim=vocab_size, output_dim=self.embedding_dim, input_length=self.seq_length),  # 词嵌入层
            layers.GlobalAveragePooling1D(),
            layers.Dense(self.hidden_dim, activation=tf.nn.relu),  # 全连接层
            layers.Dropout(rate=self.dropout_keep_prob),  # Dropout层
            layers.Dense(self.num_classes, activation=tf.nn.softmax)  # 输出层
        ])


    def trainModel(self):
        self.buildModel()
        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

        # 将文本数据转换为序列化的序号
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(self.train_content_list)
        train_sequences = tokenizer.texts_to_sequences(self.train_content_list)
        train_X = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=self.seq_length, padding='post')

        # 将标签数据转换为 one-hot 编码
        train_Y = self.label2Y(self.train_label_list)

        # 提前停止回调
        early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

        print("\nStart training model")
        start_time = time.time()

        history = self.model.fit(train_X, train_Y, epochs=self.num_iteration, batch_size=self.batch_size, verbose=2, callbacks=[early_stopping])

        duration = time.time() - start_time
        print("Time consumption: %.2f seconds" % duration)

        # Plotting training metrics
        plt.figure(figsize=(26, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.title('Training Loss')
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'])
        plt.title('Training Accuracy')
        plt.show()

        # Print training status
        for i, acc in enumerate(history.history['accuracy']):
            if i % self.print_per_batch == 0:
                print("Epoch %d ==> Loss: %.5f, Accuracy: %.4f, Time: %.2f seconds" % (i, history.history['loss'][i], acc, (duration / self.num_iteration) * i))



    # 定义预测函数
    def predict(self, content_list):
        if isinstance(content_list, str):
            content_list = [content_list]
        # 将文本数据转换为序列化的序号
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(content_list)
        sequences = tokenizer.texts_to_sequences(content_list)
        X = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=self.seq_length, padding='post')
        predict_y = self.model.predict(X)
        predict_y = np.argmax(predict_y, axis=1)
        predict_label_list = self.labelEncoder.inverse_transform(predict_y)
        return predict_label_list

    # 定义predictAll，分批预测
    def predictAll(self):
        predict_label_list = []
        batch_size = 100
        for i in range(0, len(self.test_content_list), batch_size):
            content_list = self.test_content_list[i: i + batch_size]
            predict_label = self.predict(content_list)
            predict_label_list.extend(predict_label)
        return predict_label_list

    # 打印混淆矩阵
    def printConfusionMatrix(self):
        predict_label_list = self.predictAll()
        df = pd.DataFrame(confusion_matrix(self.test_label_list, predict_label_list),
                     columns=self.labelEncoder.classes_,
                     index=self.labelEncoder.classes_)
        print('\n Confusion Matrix:')
        print(df)

    # 打印评价指标
    def printReportTable(self):
        predict_label_list = self.predictAll()
        reportTable = self.eval_model(self.test_label_list,
                                 predict_label_list,
                                 self.labelEncoder.classes_)
        print('\n Report Table:')
        print(reportTable)
        
    def eval_model(self, y_true, y_pred, labels):
        p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)
        df = pd.DataFrame(data={'Precision': p, 'Recall': r, 'F1-score': f1},
                          index=labels)
        df['Support'] = s
        return df

if __name__ == "__main__":
    print("\nStart read data")
    start_time = time.time()
    pathList, label_list = getPaths_Labels()
    content_list = getAllText(pathList)
    label_list = transform_label(label_list)
    duration = time.time() - start_time
    print("Time consumption: %.3f (min)" % (duration / 60))

    print("\nStart training model")
    start_time = time.time()
    model = TextClassification(content_list, label_list)
    model.trainModel()
    duration = time.time() - start_time
    print("Time consumption: %.3f (min)" % (duration / 60))

    model.printConfusionMatrix()
    model.printReportTable()

