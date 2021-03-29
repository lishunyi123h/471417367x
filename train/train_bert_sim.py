# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import codecs
import collections
import os
import tokenization
import tensorflow as tf
import time
from setting import Simparams
import optimization_finetuning as optimization

sim_params = Simparams()

if sim_params.model_task == 'bert':
    import bert_modeling as modeling
elif sim_params.model_task == 'albert':
    import albert_modeling as modeling
elif sim_params.model_task == 'robertta':
    import robertta_modeling as modeling
else:
    pass


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    ## 单个 训练/测试 实列的简单序列分类
    '''
    get_train_examples、get_dev_examples、get_test_examples
    这三个函数，分别对应训练、开发、测试的数据集的读取，输入参数 data_dir 由脚本启动时的 data_dir 参数获得，
    这三个函数需要把数据集每一行数据读为一个 InputExample，所有行构成 list[InputExample]
    '''

    def __init__(self, guid=None, text_a=None, text_b=None, label=None):
        """构建一个输入实列。InputExample 仅由唯一id、句子a、句子b、类标，这四个属性构成，对于单句任务，句子 b 置 None。
        Args:
          guid: 实列的唯一ID
          text_a: string.（字符串） The untokenized text of the first sequence.第一个序列的（untokenized）文本
            对于单序列任务，只能指定此序列。
          text_b: (Optional) string.（可选字符串） The untokenized text of the second sequence.序列对任务的时候指定
          label: (Optional) string.（可选字符串） 实列的标签。训练和验证集的实列需要指定，测试集不需要
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataProcessor(object):
    # 序列分类任务的数据处理基类，并实现了 XNLI、MultiNLI、MRPC、CoLA 这几个任务数据集的读取方式作为样例。
    # 仅需子类实现实现父类定义的如下四个方法
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    # def _read_tsv(cls, input_file, quotechar=None):
    #   """Reads a tab separated value file.读取以制表符分隔的值文件。"""
    #   with tf.gfile.Open(input_file, "r") as f:
    #     reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
    #     lines = []
    #     for line in reader:
    #       lines.append(line)
    #     return lines
    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[1])
                else:
                    if len(contends) == 0:
                        l = ' '.join([label for label in labels if len(label) > 0])
                        w = ' '.join([word for word in words if len(word) > 0])
                        lines.append([l, w])
                        words = []
                        labels = []
                        continue
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
            return lines


# 下面几个类继承DataProcessor
class NerProcessor(DataProcessor):
    def __init__(self):
        self.labels = set()
        output_dir = 'output_dir'
        self.output_dir = output_dir

    def get_train_examples(self, data_dir):
        data_dir = r'E:\bert-master\data'
        return self._create_example(self._read_data(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        data_dir = r'E:\bert-master\data'
        return self._create_example(self._read_data(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        data_dir = r'E:\bert-master\data'
        return self._create_example(self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self, labels=None):
        self.labels = ["O", 'B-ENT', 'I-ENT', "B-SIT", "I-SIT", "B-CON", "I-CON", "B-LOC", "I-LOC", "X", "[CLS]",
                       "[SEP]"]
        return self.labels

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            # train-0
            # 我 们 变 而 以 书 会 友 ， 以 书 结 缘 ， 把 欧 美 、 港 台 流 行 的 食 品 类 图 谱 、 画 册 、 工 具 书 汇 集 一 堂 。
            # O O O O O O O O O O O O O O O B-LOC B-LOC O B-LOC B-LOC O O O O O O O O O O O O O O O O O O O O
            examples.append(InputExample(guid=guid, text_a=text, label=label))
        return examples

    def _read_data(self, input_file):
        """Reads a BIO data."""
        self.labels = set()
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[-1])
                else:
                    if len(contends) == 0 and len(words) > 0:
                        label = []
                        word = []
                        for l, w in zip(labels, words):
                            if len(l) > 0 and len(w) > 0:
                                label.append(l)
                                self.labels.add(l)
                                word.append(w)
                        lines.append([' '.join(label), ' '.join(word)])
                        words = []
                        labels = []
                        continue
                if contends.startswith("-DOCSTART-"):
                    continue
            # [['O O O B-ORG I-ORG I-ORG O O O O O O O O O O O O O B-PER I-PER O O O O O O O O',
            # '粉 碎 “ 四 人 帮 ” 后 ， 我 家 中 的 长 辈 们 开 始 和 溥 杰 先 生 恢 复 了 联 系 。'],[]]
            return lines


class SimProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'train.tsv')
        with open(file_path, 'r', encoding="utf-8") as f:
            reader = f.readlines()
        examples = []
        for index, line in enumerate(reader):
            guid = 'train-%d' % index
            split_line = line.strip().split("\t")
            if split_line[0] == 'text_a' or len(split_line) != 3:
                continue
            text_a = tokenization.convert_to_unicode(split_line[0])
            text_b = tokenization.convert_to_unicode(split_line[1])
            label = split_line[2]
            examples.append(InputExample(guid=guid, text_a=text_a,
                                         text_b=text_b, label=label))
        return examples

    def get_dev_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'dev.tsv')
        with open(file_path, 'r', encoding="utf-8") as f:
            reader = f.readlines()
        examples = []
        for index, line in enumerate(reader):
            guid = 'train-%d' % index
            split_line = line.strip().split("\t")
            if split_line[0] == 'text_a' or len(split_line) != 3:
                continue
            text_a = tokenization.convert_to_unicode(split_line[0])
            text_b = tokenization.convert_to_unicode(split_line[1])
            label = split_line[2]
            examples.append(InputExample(guid=guid, text_a=text_a,
                                         text_b=text_b, label=label))
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""
        file_path = os.path.join(data_dir, 'test.tsv')
        with open(file_path, 'r', encoding="utf-8") as f:
            reader = f.readlines()
        examples = []
        for index, line in enumerate(reader):
            guid = 'train-%d' % index
            split_line = line.strip().split("\t")
            if split_line[0] == 'text_a' or len(split_line) != 3:
                continue
            text_a = tokenization.convert_to_unicode(split_line[0])
            text_b = tokenization.convert_to_unicode(split_line[1])
            label = split_line[2]
            examples.append(InputExample(guid=guid, text_a=text_a,
                                         text_b=text_b, label=label))
        return examples

    def get_labels(self):
        """See base class."""
        return sim_params.labels_list

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # Only the test set has a header
            if set_type == "test" and i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = tokenization.convert_to_unicode(line[2])
                label = "0"
            else:
                text_a = tokenization.convert_to_unicode(line[2])
                label = tokenization.convert_to_unicode(line[4])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    #  label_id = label_map[example.label]
    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn,
                            [per_example_loss, label_ids, logits, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features


def serving_input_fn():
    label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
    input_ids = tf.placeholder(tf.int32, [None, sim_params.max_seq_length], name='input_ids')
    input_mask = tf.placeholder(tf.int32, [None, sim_params.max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, sim_params.max_seq_length], name='segment_ids')
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'label_ids': label_ids,
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
    })()
    return input_fn


def train_bert_sim():
    start = time.time()
    tf.logging.set_verbosity(tf.logging.INFO)
    processors = {"sim": SimProcessor}

    tokenization.validate_case_matches_checkpoint(sim_params.do_lower_case, sim_params.init_checkpoint)

    if not sim_params.do_train and not sim_params.do_eval and not sim_params.do_predict:
        # 训练数据，验证数据，测试数据，至少要指定一个
        raise ValueError("At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(sim_params.bert_config_file)

    if sim_params.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (sim_params.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(sim_params.output_dir)

    task_name = sim_params.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=sim_params.vocab_file, do_lower_case=sim_params.do_lower_case)

    tpu_cluster_resolver = None
    if sim_params.use_tpu and sim_params.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            sim_params.tpu_name, zone=sim_params.tpu_zone, project=sim_params.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=sim_params.master,
        model_dir=sim_params.output_dir,
        save_checkpoints_steps=sim_params.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=sim_params.iterations_per_loop,
            num_shards=sim_params.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if sim_params.do_train:
        # 如果do_train我们定义的是True,调用get_train_examples
        train_examples = processor.get_train_examples(sim_params.data_dir)
        num_train_steps = int(len(train_examples) / sim_params.train_batch_size * sim_params.num_train_epochs)
        num_warmup_steps = int(num_train_steps * sim_params.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=sim_params.init_checkpoint,
        learning_rate=sim_params.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=sim_params.use_tpu,
        use_one_hot_embeddings=sim_params.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=sim_params.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=sim_params.train_batch_size,
        eval_batch_size=sim_params.eval_batch_size,
        predict_batch_size=sim_params.predict_batch_size)

    if sim_params.do_train:
        train_file = os.path.join(sim_params.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, sim_params.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", sim_params.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=sim_params.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if sim_params.do_eval:
        eval_examples = processor.get_dev_examples(sim_params.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if sim_params.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % sim_params.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(sim_params.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, sim_params.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", sim_params.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if sim_params.use_tpu:
            assert len(eval_examples) % sim_params.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // sim_params.eval_batch_size)

        eval_drop_remainder = True if sim_params.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=sim_params.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(sim_params.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if sim_params.do_predict:
        predict_examples = processor.get_test_examples(sim_params.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if sim_params.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % sim_params.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(sim_params.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                sim_params.max_seq_length, tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", sim_params.predict_batch_size)

        predict_drop_remainder = True if sim_params.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=sim_params.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(sim_params.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                if i >= num_actual_predict_examples:
                    break
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples

    estimator._export_to_tpu = False
    estimator.export_savedmodel(sim_params.pb_model, serving_input_fn)
    print('*' * 100)
    t = time.time() - start
    print('训练总共用时：{} min'.format(t / 60))
