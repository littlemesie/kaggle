# _*_coding:utf-8 _*_
import logging,fasttext
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#训练模型
classifier = fasttext.supervised("../../../data/smp/train.txt","model/smp.model",label_prefix="__label__")
