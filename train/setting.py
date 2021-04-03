import os
import argparse

parser = argparse.ArgumentParser()
# 下面的参数根据自己需要可以写到这里通过命令行临时改写
parser.add_argument("--TASK", type=str, default='sim', help="选择训练任务 ner intention sim squad")
opt = parser.parse_args()


class Nerparams:
    # nohup python train.py >ner 2>&1 &
    def __init__(self):
        # 寻找训练模型
        self.model_task = 'bert'  # bert albert  robertta

        self.data_dir = 'data_ner'

        model_path = 'pre_model/chinese_L-12_H-768_A-12/'
        self.bert_config_file = model_path + 'bert_config.json'
        self.vocab_file = model_path + 'vocab.txt'
        self.init_checkpoint = model_path + "bert_model.ckpt"

        self.output_dir = 'output_ner'
        self.pb_model = 'my_model_ner'

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.pb_model):
            os.makedirs(self.pb_model)

        self.labels_list = ["O", "B-ENT", "I-ENT", "B-SIT", "I-SIT", "B-CON", "I-CON",
                            "B-REG", "I-REG", "B-HHR", "I-HHR", "B-DAT", "I-DAT", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
                            "B-IND", "I-IND", "X", "[CLS]", "[SEP]"]
        self.labels_len = len(self.labels_list) + 1
        self.bi_labels = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        self.task_name = "ner"
        self.learning_rate = 2e-5
        self.max_seq_length = 128
        self.train_batch_size = 32
        self.eval_batch_size = 16
        self.predict_batch_size = 8
        self.num_train_epochs = 10.0

        self.do_lower_case = True
        self.do_train = True
        self.do_eval = True
        self.do_predict = False
        self.warmup_proportion = 0.1
        self.save_checkpoints_steps = 100
        self.iterations_per_loop = 100
        self.use_tpu = False
        self.tpu_name = None
        self.tpu_zone = None
        self.gcp_project = None
        self.master = None
        self.num_tpu_cores = 8


class Intentionparams:
    # nohup python train.py >intention 2>&1 &
    def __init__(self):
        self.model_task = 'bert'  # bert albert  robertta

        # intention意图识别
        self.data_dir = 'data_intention/'

        model_path = 'pre_model/chinese_L-12_H-768_A-12/'
        self.bert_config_file = model_path + 'bert_config.json'
        self.vocab_file = model_path + 'vocab.txt'
        self.init_checkpoint = model_path + "bert_model.ckpt"

        # model_path = 'pre_model/albert_tiny_489k/'
        # self.bert_config_file = model_path + 'albert_config_tiny.json'
        # self.vocab_file = model_path + 'vocab.txt'
        # self.init_checkpoint = model_path + "albert_model.ckpt"

        self.output_dir = 'output_intention/'
        self.pb_model = 'my_model_intention/'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.pb_model):
            os.makedirs(self.pb_model)

        self.labels_list = ['办事时效', '办事时间', '办事地点', '办事材料', '查询明细', '查询余额', '查询天气', '闲聊', '其它']
        self.max_seq_length = 128
        self.train_batch_size = 8
        self.eval_batch_size = 2
        # Adam 学习率
        self.learning_rate = 2e-6
        # 训练迭代次数
        self.num_train_epochs = 20.0

        self.predict_batch_size = 1
        self.warmup_proportion = 0.1
        self.save_checkpoints_steps = 100
        self.iterations_per_loop = 100
        self.master = None
        self.num_tpu_cores = 8
        self.do_lower_case = True


class Simparams:
    def __init__(self):
        self.model_task = 'bert'  # bert albert  robertta

        self.data_dir = 'data_sim/'

        model_path = 'pre_model/chinese_L-12_H-768_A-12/'
        self.bert_config_file = model_path + 'bert_config.json'
        self.vocab_file = model_path + 'vocab.txt'
        self.init_checkpoint = model_path + "bert_model.ckpt"

        self.output_dir = 'output_sim'
        self.pb_model = 'my_model_sim'

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.pb_model):
            os.makedirs(self.pb_model)

        self.labels_list = ["0", "1"]

        self.task_name = "sim"
        self.learning_rate = 2e-5
        self.max_seq_length = 128
        self.train_batch_size = 64
        self.eval_batch_size = 16
        self.predict_batch_size = 8
        self.num_train_epochs = 4.0

        self.do_lower_case = True
        self.do_train = True
        self.do_eval = True
        self.do_predict = False
        self.warmup_proportion = 0.1
        self.save_checkpoints_steps = 100
        self.iterations_per_loop = 100
        self.use_tpu = False
        self.tpu_name = None
        self.tpu_zone = None
        self.gcp_project = None
        self.master = None
        self.num_tpu_cores = 8


class Squadparams:
    def __init__(self):
        self.model_task = 'bert'  # bert albert  robertta

        self.data_dir = 'data_squad/cmrc2018_public/'

        model_path = 'pre_model/chinese_L-12_H-768_A-12/'
        self.bert_config_file = model_path + 'bert_config.json'
        self.vocab_file = model_path + 'vocab.txt'
        self.init_checkpoint = model_path + "bert_model.ckpt"

        # "SQuAD json for training. E.g., train-v1.1.json"
        self.train_file = 'data_squad/cmrc2018_public/' + 'train.json'
        # "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json"
        self.predict_file = 'data_squad/cmrc2018_public/' + 'dev.json'
        # 当将一个长文档分割成块时，在块之间要跨多少步。
        self.doc_stride = 128
        self.max_query_length = 64
        # The total number of n-best predictions to generate in the nbest_predictions.json output file.
        self.n_best_size = 20
        self.max_answer_length = 30
        self.master = None
        # 是否打印警告
        self.verbose_logging = False
        # 设置为True表示有些问题可以没有答案
        self.version_2_with_negative = True
        self.null_score_diff_threshold = -1.867471694946289

        self.output_dir = 'output_squad'
        self.pb_model = 'my_model_aquad'

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.pb_model):
            os.makedirs(self.pb_model)

        self.labels_list = ["0", "1"]

        self.task_name = "squad"
        self.learning_rate = 3e-5
        self.max_seq_length = 512
        self.train_batch_size = 32
        self.eval_batch_size = 8
        self.predict_batch_size = 8
        self.num_train_epochs = 2.0

        self.do_lower_case = True
        self.do_train = True
        self.do_eval = True
        self.do_predict = True
        self.warmup_proportion = 0.1
        self.save_checkpoints_steps = 100
        self.iterations_per_loop = 100
        self.use_tpu = False
        self.tpu_name = None
        self.tpu_zone = None
        self.gcp_project = None
        self.master = None
        self.num_tpu_cores = 8
        self.rand_seed = 12345
