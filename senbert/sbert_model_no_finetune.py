# -*- coding: utf-8 -*-
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from torchsummary import summary
import numpy as np
import os
import sys
local_dir = os.path.abspath('..')
sys.path.append(local_dir)
from utils import print_metrics
from data_manager import DataManager


class MyEvaluation(object):
	def __init__(self, model_name):
		self.model = SentenceTransformer(model_name)
		print(self.model)

	def evaluate(self, query_pair_list, labels):
		"""
		:param query_pair_list: [[str, str], [str, str], [str, str]...]
		:param labels: [1, 0, 1...]
		:return: pred_list: [[0.95], [0.83], [0.09]...]
		"""
		pred_list = []
		for query_pair in query_pair_list:
			embedding_pair = self.model.encode(query_pair)
			cos_sim = util.pytorch_cos_sim(embedding_pair[0], embedding_pair[1])
			score = cos_sim[0].tolist()
			pred_list.append(score)
		pred_list_ = np.array(pred_list)
		# 测试集评估打印
		print_metrics.binary_cal_metrics(labels, pred_list_)
		return pred_list

	def predict(self, query_pair_list):
		"""
		:param query_pair_list: [[str, str], [str, str], [str, str]...]
		:return: pred_list: [0.95, 0.83, 0.09...]
		"""
		pred_list = []
		for query_pair in query_pair_list:
			embedding_pair = self.model.encode(query_pair)
			cos_sim = util.pytorch_cos_sim(embedding_pair[0], embedding_pair[1])
			score = cos_sim[0].tolist()[0]
			pred_list.append(score)

		return pred_list


if __name__ == '__main__':
	# model_name = 'sentence-transformers/multi-qa-distilbert-cos-v1'
	model_name = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
	# model_name = 'sentence-transformers/all-MiniLM-L6-v2'

	# seq1_file = '../data_v4/test/test.seq1.in'
	# seq2_file = '../data_v4/test/test.seq2.in'
	# label_file = '../data_v4/test/test.label'
	# seq1_list = DataManager.load_data(seq1_file)
	# seq2_list = DataManager.load_data(seq2_file)
	# true_list = DataManager.load_labels(label_file)
	# # 手动构造query_pair_list
	# query_pair_list = []
	# for seq1, seq2 in zip(seq1_list, seq2_list):
	# 	query_pair_list.append([seq1, seq2])
	#
	# evaluater = MyEvaluation(model_name)
	# evaluater.evaluate(query_pair_list, true_list)

	data_csv_file = './用户手册and个性化闲聊_faqrank_result.csv'
	data_csv_file_new = './用户手册and个性化闲聊_faqrank_result_sbert.csv'
	col_name1 = '测试集query'
	col_name2 = 'ES Recall query'
	query_pair_list = DataManager.load_data_from_csv(data_csv_file, col_name1, col_name2)
	evaluater = MyEvaluation(model_name)
	pred_list = evaluater.predict(query_pair_list)
	# 为原data_csv_file插入结果列
	df = pd.read_csv(data_csv_file)
	df['未finetune的sbert'] = pred_list
	df.to_csv(data_csv_file_new, index=False, sep=',')

