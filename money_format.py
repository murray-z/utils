# -*- coding: utf-8 -*-

"""
@Time    : 2018/7/15 9:56
@Author  : fazhanzhang
@Function :
"""

import re


class MoneyFormat(object):
    def __init__(self, std_mat="%.2f"):
        self.format = std_mat

    def format_chinese_money_base(self, money):
        """
        格式化中文字符钱数基函数，只处理例如：五万七千零二十五元， 不能处理五十万七千零二十五元
        :param money:
        :return:
        """
        digits = {'零': 0, '壹': 1, '贰': 2, '貳': 2, '叁': 3, '肆': 4, '伍': 5, '陆': 6, '陸': 6, '柒': 7, '捌': 8, '玖': 9,
                  '一': 1, '二': 2, '两': 2,  '三': 3,  '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
        units = {'十': 10, '拾': 10, '百': 100, '佰': 100, '千': 1000, '仟': 1000, '万': 10000, '亿': 100000000,
                 '角': 0.1, '分': 0.01}
        temp = []
        sub_money = []
        for _ in money:
            digit = digits.get(_, 0)
            if digit != 0:
                temp.append(digit)
            unit = units.get(_, 0)

            if digit == 0 and unit != 0:
                if not temp:
                    sub_money.append(0)
                else:
                    sub_money.append(temp[-1]*unit)
        return sum(sub_money)

    def format_chinese_money(self, money):
        """
        处理包含“万”， “亿”的字符串
        :param money:
        :return:
        """
        if '亿' in money:
            money_subs = re.split("亿", money, maxsplit=1)
            return self.format_chinese_money_base(money_subs[0]+'元') * 100000000 + self.format_chinese_money(money_subs[1])
        if '万' in money or '萬' in money:
            money_subs = re.split("万", money, maxsplit=1)
            return self.format_chinese_money_base(money_subs[0]+'元') * 10000 + self.format_chinese_money_base(money_subs[1])
        return self.format_chinese_money_base(money)

    def format_alabo_money(self, money):
        money = re.sub("[,，]", "", money)
        if money[-2:] == "万元":
            return float(money[:-2])*10000
        if money[-2:] == "亿元":
            return float(money[:-2])*100000000
        return float(money[:-1])

    def money_format(self, money, std_mat="%.2f"):
        if re.match("\d+", money):
            return std_mat % self.format_alabo_money(money)+'元'
        else:
            return std_mat % self.format_chinese_money(money)+'元'


if __name__ == '__main__':
    money_format = MoneyFormat()
    print(money_format.money_format("一千八百万元"))
    # print(money_format.format_chinese_money_base('一千八百元整'))