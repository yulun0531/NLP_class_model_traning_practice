from gensim.models import Word2Vec
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import reuters

# 加載reuters新聞標題
def load_reuters_data():
    X = []
    y = []
    for file_id in reuters.fileids():
        category = reuters.categories(file_id)[0]
        title = ' '.join(reuters.words(file_id))
        X.append(title)
        y.append(category)
    return X, y

X, y = load_reuters_data()

# 將文本轉換為詞向量
def text_to_vector(text, model):
    words = text.split()
    vectors = []
    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# 加載Word2Vec模型
model = Word2Vec.load("word2vec_model.bin")

# 轉換文本為詞向量
X_vectors = [text_to_vector(text, model) for text in X]

# 切分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

# 使用邏輯回歸模型進行分類
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

# 在測試集上進行預測
y_pred = classifier.predict(X_test)

# 計算準確率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
