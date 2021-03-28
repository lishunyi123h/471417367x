import numpy as np


class IntentionBert:
    def __init__(self, Intentionparams):
        self.intentionbert = Intentionparams()

    def convert_single_example(self, example, max_seq_length, tokenizer):
        textlist = example.text.split(' ')
        tokens = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
        ntokens.append("[SEP]")
        segment_ids.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            ntokens.append("**NULL**")
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        feature = InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=0)

        return feature

    def query_examples(self, query, guid='text-0'):
        # 如果用户输入问句中有空格就用‘-’替换，不然textlist = example.text.split(' ')这里会切分出来两个空格元素
        query = query.replace(' ', '-')
        text = ' '.join(query)
        predict_examples = InputExample(guid=guid, text=text, label='其他')
        return predict_examples

    def intention_predict(self, query):
        feature = self.convert_single_example(self.query_examples(query), self.intentionbert.max_seq_length,
                                              self.intentionbert.tokenizer)
        prediction = self.intentionbert.predict_fn({
            "input_ids": [feature.input_ids],
            "input_mask": [feature.input_mask],
            "segment_ids": [feature.segment_ids],
            "label_ids": [feature.label_ids]
        })
        score = np.max(prediction['probabilities'])
        index = np.argmax(prediction['probabilities'])
        classification = self.intentionbert.label_list[int(index)]

        return classification, score


class InputExample(object):
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
