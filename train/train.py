from train_bert_ner import train_bert_ner
from train_bert_intention import train_bert_intention
from train_bert_sim import train_bert_sim
from train_bert_squad import train_bert_squad
from setting import opt
train = {"ner": train_bert_ner,
         "intention": train_bert_intention,
         "sim": train_bert_sim,
         "squad": train_bert_squad}
train[opt.TASK]()
