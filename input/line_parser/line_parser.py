from __future__ import absolute_import

import tensorflow as tf

class DefaultLinerParser():
	def parse(self, value):
		file_name, label = tf.decode_csv(value, [[''], [0]])
		return file_name, label
