import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report


# 加載數據，跳過包含錯誤格式的行
columns_to_read = ['ItemID', 'Sentiment', 'SentimentSource', 'SentimentText']

# 加載數據，只讀取指定的列
data = pd.read_csv("data.csv", usecols=columns_to_read, nrows=10000)
X = data['SentimentText']
y = data['Sentiment']

# 將數據劃分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用TF-IDF特徵提取
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 初始化支持向量機分類器
svm_classifier = SVC(kernel='linear', verbose=True)

# 在訓練集上訓練模型
svm_classifier.fit(X_train_tfidf, y_train)

# 在測試集上進行預測
predictions = svm_classifier.predict(X_test_tfidf)

# 輸出分類報告
print(classification_report(y_test, predictions))