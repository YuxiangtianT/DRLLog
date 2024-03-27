from gensim.models import word2vec

vocabs = []
max_len = 0
with open('data/hdfs_train', 'r') as f:
    lines = f.readlines()
    for line in lines:
        vocabs.append(line)
        if len(line) > max_len:
            max_len = len(line)

# 生成词向量空间模型
model = word2vec.Word2Vec(sentences=vocabs, sg=1, vector_size=100,  window=5,  min_count=1,  negative=3, sample=0.001, hs=1, workers=4)

model.save('w2c.model')

# 加载模型
new_model = word2vec.Word2Vec.load('w2c.model')

new_model.train(vocabs, total_examples=4855, epochs=10)

new_model.save('w2c.model')

# 获取所有单词的向量表示
all_word_vectors = model.wv.vectors

# 获取所有单词对应的词汇表
all_words = model.wv.index_to_key

# 打印单词和对应的向量表示
for word, vector in zip(all_words, all_word_vectors):
    print(f"{word}，1: {vector}")

