from train_bert_ner import train_bert_ner
from train_bert_intention import train_bert_intention
from train_bert_sim import train_bert_sim
from setting import opt

if opt.TASK == 'ner':
    train_bert_ner()
elif opt.TASK == 'intention':
    train_bert_intention()
elif opt.TASK == 'sim':
    train_bert_sim()
else:
    print('TASK:{}, 只能是“ner”，“intention”，“sim”'.format(opt.TASK))
