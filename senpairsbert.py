import tokenization


class SenpairsBert:
    def __init__(self, Senpairsparams):
        self.senpairsbert = Senpairsparams()

    def query_examples(self, text_a, text_b, guid='text-0'):
        return InputExample(guid=guid, text_a=text_a, text_b=text_b, label='0')

    def get_token(self, text):
        return tokenization.convert_to_unicode(text)

    def convert_single_example(self, example, label_list, max_seq_length, tokenizer):
        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i
        #  label_id = label_map[example.label]
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

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

        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        label_id = label_map[example.label]

        feature = InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_id,
                                is_real_example=True)
        return feature

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def senpairs_predict(self, text_a, text_b):
        predict_example = self.query_examples(text_a, text_b)
        feature = self.convert_single_example(predict_example, self.senpairsbert.labels_list,
                                              self.senpairsbert.max_seq_length, self.senpairsbert.tokenizer)

        prediction = self.senpairsbert.senpairs_model({
            "input_ids": [feature.input_ids],
            "input_mask": [feature.input_mask],
            "segment_ids": [feature.segment_ids],
            "label_ids": [feature.label_id]
        })
        return prediction['probabilities'][0][1]


class InputExample(object):
    def __init__(self, guid=None, text_a=None, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id, is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


if __name__ == "__main__":
    from setting import Senpairsparams
    import time

    simp = SenpairsBert(Senpairsparams)

    # answers = ['申领护照在哪了办理', '需要什么材料办理身份证', '是在派出所办身份证吗', '临时身份证办理需要什么条件',
    #            '办理身份证的材料丢了怎么补办', '护照办理要什么材料', '居住证办理需要什么材料', '在哪里办理居住证']
    answers = ['我要办理身份证', '我想开饭店', '我要开公司', '开饭店需要什么条件', '饭店营业执照要过期了怎么办',
               '我想变更公司法人', '开饭店需要的材料怎么预审', '开饭店需要做消防备案吗', '申领火锅店的营业执照']
    answers_token = [simp.get_token(a) for a in answers]

    while True:
        # query = '身份证办理需要什么材料'
        query = input('请输入你的问句：')
        start = time.time()

        query_token = simp.get_token(query)

        for at in range(len(answers_token)):
            label = simp.senpairs_predict(query_token, answers_token[at])
            if label > 0.5:
                print('相似句：{}   {:.3}'.format(answers[at], label))
            else:
                print('==不相似句：{}   {:.3}'.format(answers[at], label))
        print('花费时间：{}'.format(time.time() - start))
        print('=' * 50)
