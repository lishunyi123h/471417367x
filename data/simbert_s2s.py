import numpy as np
from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.snippets import sequence_padding
from bert4keras.snippets import AutoRegressiveDecoder


class Sengenerate:
    def __init__(self):
        self.maxlen = 128
        self.config_path = 'chinese_simbert_L-12_H-768_A-12/bert_config.json'
        self.checkpoint_path = 'chinese_simbert_L-12_H-768_A-12/bert_model.ckpt'
        self.dict_path = 'chinese_simbert_L-12_H-768_A-12/vocab.txt'

        # 加载并精简词表，建立分词器
        self.token_dict, self.keep_tokens = load_vocab(
            dict_path=self.dict_path,
            simplified=True,
            startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
        )

        self.tokenizer = Tokenizer(self.token_dict, do_lower_case=True)

        # 建立加载模型
        self.bert = build_transformer_model(
            self.config_path,
            self.checkpoint_path,
            with_pool='linear',
            application='unilm',
            keep_tokens=self.keep_tokens,  # 只保留keep_tokens中的字，精简原字表
            return_keras_model=False,
        )

        self.encoder = keras.models.Model(self.bert.model.inputs, self.bert.model.outputs[0])
        self.seq2seq = keras.models.Model(self.bert.model.inputs, self.bert.model.outputs[1])


sg = Sengenerate()


class SynonymsGenerator(AutoRegressiveDecoder):
    @AutoRegressiveDecoder.set_rtype('probas')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return sg.seq2seq.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, n=1, topk=5):
        token_ids, segment_ids = sg.tokenizer.encode(text, max_length=sg.maxlen)
        output_ids = self.random_sample([token_ids, segment_ids], n, topk)  # 基于随机采样
        return [sg.tokenizer.decode(ids) for ids in output_ids]


synonyms_generator = SynonymsGenerator(start_id=None, end_id=sg.tokenizer._token_end_id, maxlen=sg.maxlen)


def gen_synonyms(text, n=100, k=20):
    r = synonyms_generator.generate(text, n)
    r = [i for i in set(r) if i != text]
    r = [text] + r
    X, S = [], []
    for t in r:
        x, s = sg.tokenizer.encode(t)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = sg.encoder.predict([X, S])
    Z /= (Z ** 2).sum(axis=1, keepdims=True) ** 0.5
    argsort = np.dot(Z[1:], -Z[0]).argsort()
    for i in argsort[:k]:
        print(r[i + 1])


if __name__ == '__main__':
    while True:
        text = input('输入：')
        gen_synonyms(text)
        print('=' * 50)
