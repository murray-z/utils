# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
from wordcloud import WordCloud


class WordsCloud:
    def __init__(self, font_path="./data/simfang.ttf", mask_path="./data/apple.png", save_path="./wordcloud.png",
                 with_mask=False):
        """
        画词云
        :param font_path: 字体路径
        :param mask_path: 背景图片路径
        :param save_path: 生成词云保存路径
        :param with_mask: 是否使用背景，默认不使用
        """
        self.font_path = font_path
        self.mask_path = mask_path
        self.save_path = save_path
        self.with_mask = with_mask

    def wordcloud(self, keyphrrase_weight):
        if self.with_mask:
            mask = np.array(Image.open(self.mask_path))
            wc = WordCloud(
                background_color='white',
                width=800,
                height=800,
                mask=mask,
                font_path=self.font_path,
            )
        else:
            wc = WordCloud(
                background_color='white',
                width=800,
                height=800,
                font_path=self.font_path,
            )
        wc.generate_from_frequencies(keyphrrase_weight)  # 绘制图片
        wc.to_file(self.save_path)  # 保存图片


if __name__ == "__main__":
    word_cloud = WordsCloud()
    word_cloud.wordcloud({"成都": 50, "北京": 30})