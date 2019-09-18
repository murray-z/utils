# -*- coding: utf-8 -*-

"""
由于谷歌提供的bert模型的预测代码，只能输入文件，而且每次预测都得重新加载模型，
这里我们尝试修改原模型，可以将模型参数加载到内存中。

以文本分类为例：
run_classifier.py
"""


"""
step1: 转换模型

1. 将文件中函数main(_)中的 if FLAGS.do_predict: 下面的代码注释掉，重新编写
2. 编写转换代码，如下：

if FLAGS.do_predict:
    my_bert_model_path = "./my_bert_model"
    if not os.path.exists(my_bert_model_path):
        os.mkdir(my_bert_model_path)

    def serving_input_fn():
        label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
        input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_ids')
        input_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_mask')
        segment_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='segment_ids')
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'label_ids': label_ids,
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
        })()
        return input_fn


    estimator._export_to_tpu = False
    estimator.export_savedmodel(my_bert_model_path, serving_input_fn)

3. 执行代码，新建run.py，

BERT_BASE_DIR = ""       # bert_base路径
data_dir = ""            # data路径
task_name = ""           # 任务名称
max_seq_length = 128     # 序列长度
TRAINED_CLASSIFIER = ""  # fine-tune生成模型路径
output_dir = ""          # fine-tune生成模型路径

os.system("python run_classifier.py \
      --task_name={} \
      --do_predict=true \
      --data_dir={} \
      --vocab_file={}/vocab.txt \
      --bert_config_file={}/bert_config.json \
      --init_checkpoint={}/model.ckpt \
      --max_seq_length={} \
      --output_dir={}".format(task_name, data_dir, BERT_BASE_DIR, BERT_BASE_DIR,
                          TRAINED_CLASSIFIER, max_seq_length, output_dir))

4. python run.py
5. 在 ./my_bert_model 目录下生成时间戳文件夹，转换后模型存放在该文件夹下
"""


"""
step2：利用转换后模型进行预测


在run_classifier.py 下新建类 BertPredict


Args:
    my_bert_model_path: step1生成模型路径
    vocab_path：词典文件路径
    label_path: 标签文件路径
    batch_size: batch size
    max_seq_len: 最大序列长度


class BertPredict():
    def __init__(self, my_bert_model_path, vocab_path, label_path, batch_size, max_seq_len):
        self.predict_fn = tf.contrib.predictor.from_saved_model(my_bert_model_path)
        self.label_list = self.load_json(label_path)
        self.batch_size = batch_size
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=False)
        self.max_seq_len = max_seq_len

    def load_json(self, json_path):
        with open(json_path) as f:
            return json.loads(f.read())

    def predict(self, input_list):
        result = []
        examples = [InputExample("id-{}".format(idx), item, None, '01') for idx, item in enumerate(input_list)]
        all_features = [convert_single_example(100, example, self.label_list, self.max_seq_len, self.tokenizer)
                    for example in examples]

        num_batchs = int(len(examples)/self.batch_size) \
            if int(len(examples)/self.batch_size) == len(examples)/self.batch_size \
            else int(len(examples)/self.batch_size)+1

        for i in range(num_batchs):
            features = all_features[i*self.batch_size: (i+1)*self.batch_size]

            prediction = self.predict_fn({
                "input_ids": [feature.input_ids for feature in features],
                "input_mask": [feature.input_mask for feature in features],
                "segment_ids": [feature.segment_ids for feature in features],
                "label_ids": [feature.label_id for feature in features],
            })

            probabilities = prediction["probabilities"]
            label = [self.label_list[idx] for idx in probabilities.argmax(axis=1).tolist()]
            result.extend(label)
        return result

"""


