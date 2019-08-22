# -*- coding: utf-8 -*-


import kenlm


def train_lm():
    """
    step1: /kenlm/build/bin/lmplz -o 3 --verbose_header --text lm_model.txt --arpa lm_model.arps
    step2: /kenlm/build/bin/build_binary lm_model.arps lm_model.klm
    :return:
    """
    pass


def test(lm_model_dir='./'):
    lm_model = kenlm.Model(lm_model_dir)
    score = lm_model.score("小明，早上好！")