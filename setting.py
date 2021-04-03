import tensorflow as tf
import tokenization
import numpy as np
from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import open
import json
import pandas as pd


class Nerparams:
    def __init__(self):
        self.max_seq_length = 128
        self.ner_model = tf.contrib.predictor.from_saved_model('data/ner_model')
        self.label_list = ["O", "B-ENT", "I-ENT", "B-SIT", "I-SIT", "B-CON", "I-CON", "B-REG", "I-REG", "B-HHR",
                           "I-HHR", "B-DAT", "I-DAT", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-IND", "I-IND", "B-CSW",
                           "I-CSW", "B-GSX", "I-GSX", "B-XSC", "I-XSC", "X", "[CLS]",
                           "[SEP]"]
        self.id2label = {}
        for x, y in enumerate(self.label_list):
            self.id2label[str(x)] = y
        self.label_map = {}
        for (i, label) in enumerate(self.label_list, 1):
            self.label_map[label] = i
        self.tokenizer = tokenization.FullTokenizer(vocab_file='data/vocab.txt', do_lower_case=True)


class Intentionparams:
    def __init__(self):
        self.max_seq_length = 128
        self.predict_fn = tf.contrib.predictor.from_saved_model('data/intention_model')
        self.label_list = ['办事时效', '办事时间', '办事地点', '办事材料', '查询明细', '查询余额', '查询天气', '闲聊', '其它']
        self.tokenizer = tokenization.FullTokenizer(vocab_file='data/vocab.txt', do_lower_case=True)


class Simparams:
    def __init__(self):
        self.max_seq_length = 128
        self.corpus_text = 'data/corpus3.json'
        self.config_path = 'data/chinese_simbert_L-12_H-768_A-12/bert_config.json'
        self.checkpoint_path = 'data/chinese_simbert_L-12_H-768_A-12/bert_model.ckpt'
        self.vocab_file = 'data/chinese_simbert_L-12_H-768_A-12/vocab.txt'
        self.tokenizer = Tokenizer(self.vocab_file, do_lower_case=True)  # 建立分词器

        # 加载模型
        self.bert = build_transformer_model(
            self.config_path,
            self.checkpoint_path,
            with_pool='linear',
            application='unilm',
            return_keras_model=False,
        )

        self.encoder = keras.models.Model(self.bert.model.inputs, self.bert.model.outputs[0])
        # 加载数据库语料
        with open(self.corpus_text, 'r', encoding='utf-8') as load_f:
            self.classes = json.load(load_f)

        self.corpus = eval(self.classes)
        self.list_vec = []
        self.list_corpus = []
        for c, v in self.corpus.items():
            self.list_vec.append(v)
            self.list_corpus.append(c)
        # 新增语料
        df = pd.read_excel('data/新增数据.xlsx')
        for vn in range(len(df)):
            self.list_corpus.append(df['新增语料'][vn])
            self.list_vec.append(self.vec(df['新增语料'][vn]))

        self.list_vec = np.concatenate(self.list_vec).reshape(-1, 768)

    def vec(self, query):
        token_ids, segment_ids = self.tokenizer.encode(query, max_length=self.max_seq_length)
        vec = self.encoder.predict([[token_ids], [segment_ids]])[0]
        # 求单位向量
        vec /= (vec ** 2).sum() ** 0.5
        return vec


class Senpairsparams:
    def __init__(self):
        self.max_seq_length = 128
        self.senpairs_model = tf.contrib.predictor.from_saved_model('data/senpairs_model')

        self.labels_list = ["0", "1"]
        self.tokenizer = tokenization.FullTokenizer(vocab_file='data/vocab.txt', do_lower_case=True)
