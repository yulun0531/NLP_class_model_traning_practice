from gensim.models import Word2Vec
import os

# 測試類比函數
def test_analogy(model, test_data):
    for word_a, word_b, expected_word_c, correct_predicted_word_c in test_data:
        predicted_word_c = model.wv.most_similar(positive=[word_b, expected_word_c], negative=[word_a], topn=1)[0][0]
        print(f"{word_a}:{word_b} :: {expected_word_c}:{predicted_word_c} {'[CORRECT]' if predicted_word_c == correct_predicted_word_c else '[INCORRECT]'}")

# BATS資料集所在的主目錄路徑
main_dir = 'BATS_3.0'

# 獲取BATS資料集中所有子資料夾
subdirs = [d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]

# 用來存放所有句子的列表
sentences = []

# 逐個讀取每個子資料夾中的所有txt檔案
for subdir in subdirs:
    subdir_path = os.path.join(main_dir, subdir)
    
    # 獲取子資料夾中所有的txt檔案
    txt_files = [f for f in os.listdir(subdir_path) if f.endswith('.txt')]
    
    # 逐個讀取txt檔案中的內容
    for txt_file in txt_files:
        txt_file_path = os.path.join(subdir_path, txt_file)
        with open(txt_file_path, 'r') as f:
            # 逐行讀取資料
            for line in f:
                # 分割每行資料成為 "backward" 和 "forward/forwards/frontward/frontwards/forrad/forrard/forth/onward"
                backward, forward_options = line.strip().split('\t')
                # 將 forward 選項分割成單詞
                forward_options = forward_options.split('/')
                # 將每個前向選項添加到句子列表中
                for forward_option in forward_options:
                    sentence = [backward, forward_option]
                    sentences.append(sentence)

# 訓練Word2Vec模型
model = Word2Vec(sentences, vector_size=200, window=25, min_count=1, workers=8, epochs=250)
model.save("word2vec_model.bin")

# 測試模型的類比預測
test_data = [("expect", "expected", "follow", "followed")]
test_data1 = [("allowing", "allows", "learning", "learns")]
test_data2 = [("adds", "added", "allows", "allowed")]
test_analogy(model, test_data)
test_analogy(model, test_data1)
test_analogy(model, test_data2)


	