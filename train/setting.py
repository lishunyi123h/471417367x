import os

# ner intention
TASK = 'ner'


class Nerparams:
    # nohup python albert_ner.py >ner 2>&1 &
    def __init__(self):
        self.model_task = 'bert'  # albert  robertta

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
    # nohup python albert_intention.py >intention 2>&1 &
    def __init__(self):
        self.model_task = 'albert'  # 'albert

        # intention意图识别
        self.data_dir = 'data_intention/'
        # 模型训练结果的检查点文件保存路径
        # model_path = 'chinese_L-12_H-768_A-12/'
        # self.bert_config_file = model_path + 'bert_config.json'
        # self.vocab_file = model_path + 'vocab.txt'
        # self.init_checkpoint = model_path + "bert_model.ckpt"

        model_path = 'pre_model/albert_tiny_489k/'
        self.bert_config_file = model_path + 'albert_config_tiny.json'
        self.vocab_file = model_path + 'vocab.txt'
        self.init_checkpoint = model_path + "albert_model.ckpt"

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
        self.num_train_epochs = 5.0

        self.predict_batch_size = 1
        self.warmup_proportion = 0.1
        self.save_checkpoints_steps = 100
        self.iterations_per_loop = 100
        self.master = None
        self.num_tpu_cores = 8
        self.do_lower_case = True
