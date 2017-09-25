from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
import tensorflow as tf

sys.path.insert(0,'../places365-tf/slim')
from preprocessing import preprocessing_factory


class ImageClassifier(object):
    def __init__(self,
                 model_name='shufflenet_50_g4_d272',
                 label_file='./models/indoorCVPR_09_labels_noindx.txt',
                 graph_file='./models/shufflenet_50_g4_d272_indoorCVPR09.pb',
                 graph_output_node='shufflenet_50/predictions/Softmax'):
        super(ImageClassifier, self).__init__()

        # Parameters
#        self.model_name = 'shufflenet'
#        self.data_set = 'indoorCVPR_09'
        #self.label_file = './models/indoorCVPR_09_labels_noindx.txt'
        #self.graph_file = './models/shufflenet_50_g4_d272_indoorCVPR09.pb'
        self.model_name = model_name
        self.output_node = graph_output_node

        self.labels = []

        self.load_labels(label_file)
        self.create_graph(graph_file)
        self.sess = tf.Session()

    def load_labels(self, label_file):
        proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
        for l in proto_as_ascii_lines:
            self.labels.append(l.rstrip())

    def create_graph(self, graph_file):
        """Creates a graph from saved GraphDef file and returns a saver."""
        # Creates graph from saved graph_def.pb.
        with tf.gfile.FastGFile(graph_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    def run_inference_on_image(self, image):
        """Runs inference on an image.
        Args:
        image: Image file name.
        Returns:
        Nothing
        """

        # image_data = tf.gfile.FastGFile(image, 'rb').read()

        image = tf.image.decode_jpeg(tf.read_file(image),
                                     channels=3)

        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            self.model_name, is_training=False)
        processed_image = image_preprocessing_fn(image, 224, 224)

        processed_images = tf.expand_dims(processed_image, 0)
        # sess = tf.Session()
        # im_result = sess.run(processed_images)

        # Creates graph from saved GraphDef.
        # create_graph()

        # with tf.Session() as sess:
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.

        # names = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        # print(names[-5:])

        im_result = self.sess.run(processed_images)

        input_tensor = self.sess.graph.get_operation_by_name('input')
        output_tensor = self.sess.graph.get_operation_by_name(self.output_node)
        predictions = self.sess.run(output_tensor.outputs[0],
                               {input_tensor.outputs[0]: im_result})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-5:][::-1]
        # labels = load_labels(FLAGS.label_file)
        result = {}
        labels = []
        scores = []
        for node_id in top_k:
            human_string = self.labels[node_id]
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))
            labels.append(human_string)
            scores.append(score)
        result['top5_labels'] = labels
        result['top5_scores'] = scores

        return result
