from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import input.line_parser as line_parser
import input.data_reader as data_reader

_FILE_LIST_ = ['./data/data']

parser = line_parser.DefaultLinerParser()

batch_data, batch_label = data_reader.read_data(_FILE_LIST_, line_parser = parser)

with tf.Session() as session:
	session.run(tf.global_variables_initializer())
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord = coord)

	try:
		#while not coord.should_stop():                                                                                                                                                                                                                                       
		while True:
			example, label = session.run([batch_data, batch_label])
			print example
	except tf.errors.OutOfRangeError:
		print 'Done reading'
	finally:
		coord.request_stop()

	coord.join(threads)
	sess.close()