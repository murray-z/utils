# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class TopicParse(object):
    """
    主题发现
    """
    def __init__(self, train_data, n_components=10, n_top_words=50):
        """
        :param train_data: 训练数据

                      格式：   ["张三 在 中国移动 工作", "你 是 谁 ？"]


        :param n_components:  主题数目

        :param n_top_words:  每个主题提取的主题词数目

        """
        self.train_data = train_data
        self.n_components = n_components
        self.n_top_words = n_top_words

    def print_top_words(self, model, feature_names, n_top_words):
        ret_str = ''
        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            message += " ".join([feature_names[i]
                                 for i in topic.argsort()[:-n_top_words - 1:-1]])
            ret_str += message+'\n'
        print(ret_str)
        return ret_str

    def lda(self):
        tf_vectorizer = CountVectorizer()

        tf = tf_vectorizer.fit_transform(self.train_data)

        lda = LatentDirichletAllocation(n_components=self.n_components, max_iter=50,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)
        lda.fit(tf)
        tf_feature_names = tf_vectorizer.get_feature_names()
        return self.print_top_words(lda, tf_feature_names, self.n_top_words)


if __name__ == '__main__':
    topic_parse = TopicParse(["张三 在 中国移动 工作", "你 是 谁 ？"])
    topic_words = topic_parse.lda()