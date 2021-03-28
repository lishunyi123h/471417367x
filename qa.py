from nerbert import NerBert
from intentionbert import IntentionBert
from simbert import SimBert
from setting import Nerparams, Intentionparams, Simparams
import time

s = time.time()
ner = NerBert(Nerparams)
intention = IntentionBert(Intentionparams)
sim = SimBert(Simparams)
print('加载时间：{}'.format(time.time() - s))

while True:
    print("=" * 80)
    query = input("输入：")
    start = time.time()

    ner_re = ner.ner_predict(query)
    classification, score = intention.intention_predict(query)
    print('意图：{}，得分：{}'.format(classification, score))
    print('关键词：{}'.format(ner_re))
    print('数据库相似度句子匹配')
    sim.most_similar(query, topn=10)
    print('时间：{}'.format(time.time() - start))
