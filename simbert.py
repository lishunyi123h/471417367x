import numpy as np
from setting import Simparams


class SimBert:
    def __init__(self, Simparams):
        self.simbert = Simparams()

    def most_similar(self, query, topn=10):
        vec1 = self.simbert.vec(query)
        sims = np.dot(self.simbert.list_vec, vec1)
        for j in np.array(sims).argsort()[::-1][:topn]:
            print('相似度{:.3}： {}'.format(sims[j], self.simbert.list_corpus[j]))


if __name__ == "__main__":
    simb = SimBert(Simparams)
    while True:
        text = input('请问有什么可以帮助您的：')
        simb.most_similar(text)
        print('=' * 50)
