class NerBert:
    def __init__(self, Nerparams):
        self.nerbert = Nerparams()

    def query_examples(self, query, guid='text-0'):
        # 如果用户输入问句中有空格就用‘-’替换，不然textlist = example.text.split(' ')这里会切分出来两个空格元素
        query = query.replace(' ', '-')
        label = ''
        for _ in query:
            label = label + 'O'
        label = ' '.join(label)
        text = ' '.join(query)
        predict_examples = [InputExample(guid=guid, text=text, label=label)]
        return predict_examples

    def convert_single_example(self, example, label_map, max_seq_length, tokenizer):
        textlist = example.text.split(' ')
        labellist = example.label.split(' ')
        tokens = []
        labels = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("X")
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            ntokens.append("**NULL**")
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        feature = InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                                label_ids=label_ids)

        return feature

    def convert_id_label(self, text, predict, id2label):
        # 只保留识别出的ner元素
        predict = predict["output"][0]
        text = ' ' + text + ' '
        ner_reg_list = []
        for word, tag in zip(text, predict):
            tag = id2label.get(str(int(tag) - 1), "O")
            ner_reg_list.append((word, tag))
        ners = {}
        cur_flag = None
        cur_ner = []
        cur_type = None
        for word, tag in ner_reg_list:
            # print(word, tag, cur_flag, cur_ner)
            # 标注词结束
            if cur_flag is not None and tag != cur_flag:
                if cur_type is not None:
                    cur_type_lower = cur_type.lower()
                    if cur_type_lower not in ners:
                        ners[cur_type_lower] = []
                    ners[cur_type_lower].append(''.join(cur_ner))
                    cur_flag = None
                    cur_ner.clear()
                    cur_type = None

            # 标注词开始
            if tag.startswith('B-'):
                cur_type = tag.split("-")[1]
                cur_flag = "I-" + cur_type
                cur_ner.clear()
                cur_ner.append(word)
                continue

            # 标注词中部
            if tag == cur_flag:
                cur_ner.append(word)

        return ners

    def ner_predict(self, query):
        predict_example = self.query_examples(query)[0]
        feature = self.convert_single_example(predict_example, self.nerbert.label_map, self.nerbert.max_seq_length,
                                              self.nerbert.tokenizer)

        prediction = self.nerbert.ner_model({
            "input_ids": [feature.input_ids],
            "input_mask": [feature.input_mask],
            "segment_ids": [feature.segment_ids],
            "label_ids": [feature.label_ids]
        })
        re_ner = self.convert_id_label(query, prediction, self.nerbert.id2label)
        return re_ner


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
