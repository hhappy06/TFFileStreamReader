from __future__ import absolute_import

import tensorflow as tf
from input.line_parser.line_parser import DefaultLinerParser

_BATCH_SIZE_ = 2

def read_data(file_list, line_parser = DefaultLinerParser, num_epochs = None):
	file_name_queue = tf.train.string_input_producer(file_list, num_epochs = num_epochs)
	stream_reader = tf.TextLineReader()
	key,value = stream_reader.read(file_name_queue)
	feature, label = line_parser.parse(value)
	min_after_dequeu = 2
	num_threads = 2
	capacity = min_after_dequeu + _BATCH_SIZE_ * num_threads
	batch_data, batch_label = tf.train.shuffle_batch([feature, label],
		batch_size = _BATCH_SIZE_,
		capacity = capacity, 
		min_after_dequeue = min_after_dequeu)
	return batch_data, batch_label
