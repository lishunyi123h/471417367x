from train_bert_ner import train_bert_ner
from train_bert_intention import train_bert_intention
from setting import TASK

if TASK == 'ner':
    train_bert_ner()
elif TASK == 'intention':
    train_bert_intention()
else:
    print('TASK:{}, 只能是“ner”，“intention”'.format(TASK))
