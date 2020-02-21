from PIL import Image
from collections import Counter
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import boto3
import json
import os

def model(x_train,y_train,x_test,y_test,max_caption_length,emb_layer,top_words):
	"""Generating a model for this task"""
	## getting the image processing side of the model ready ##

	visionmodel = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet',input_tensor=None,
			input_shape=(299,299,3), pooling='max')
	imageinput = visionmodel.input
	imageoutput = visionmodel.layers[-2].output
	
	## now lets get the text processing part ready
	
	seq_input = tf.keras.layers.Input(shape=(maxlength,),dtype='int32')
		## We should pass in the embedded layer
	#self.embedded_layer = tf.keras.layers.Embedding(top_words,embedding_dim,
			embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
			input_length=maxlength,trainable=False)
	self.embedded_layer = emb_layer ## We get control here of what kind of embeddings we want to use
	embedded_sequences = embedded_layer(seq_input)
	encoded = tf.keras.layers.LSTM(256,return_sequences=True,dropout=0.2,
			kernel_regularizer=tf.keras.regularizers.l2(0.000010))(embedded_sequences)
	encoded1 = tf.keras.layers.LSTM(128,return_sequences=True,dropout=0.2,
			kernel_regularizer=tf.keras.regularizers.l2(0.000010))(encoded)
	encoded2 = tf.keras.layers.LSTM(64,dropout=0.2,kernel_regularizer=tf.keras.regularizers.l2(0.000010))(encoded1)
	concatop = tf.keras.layers.concatenate([imageoutput,encoded2],axis=1)

	## Now for the MLP
	out = tf.keras.layers.Dense(32768,activation='relu')(concatop)
	out1 = tf.keras.layers.Dense(16384,activation='relu')(out)
	out2 = tf.keras.layers.Dense(9000,activation='softmax')(out1)

	main_input1 = tf.keras.layers.Input(shape=(299, 299, 3,),dtype='float32')
	main_input2 = tf.keras.layers.Input(shape=(maxlength,),dtype='int32')

	classifier = tf.keras.models.Model([visionmodel.input,seq_input],outputs=out2)
	
	classifier.compile(optimizer='RMSprop',loss='categorical_crossentropy',
			metrics=['accuracy'])
	classifier.fit(x_train,y_train) ## We should set up fit to save the best model
	classifier.evaluate(x_test,y_test)

	return classifier

def _load_training_data(base_dir):
	"""Load COCO dataset training data"""

def _load_testing_data(base_dir):
	"""Load COCO dataset testing data"""

def _create_embedding_layer():
	"""Creates and returns an embedding layer"""


def _parse_args():
	parser = argparse.ArgumentParser()
	
	# Data, model, and output directories
	# model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.

	parser.add_argument('--model_dir', type=str)
	parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
	parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
	parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TESTING'))
	parser.add_argument('--topwords', type=str, default=9000)
	parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
	parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

	return parser.parse_known_args()

if __name__ == "__main__":
	args, unknown = _parse_args()

	train_data,train_labels = _load_training_data(args.train)
	eval_data,eval_labels = _load_testing_data(args.test)
	emb_layer = _create_embedding_layer()

	captioning_classifier = model(train_data,train_labels,eval_data,eval_labels,emb_layer,args.topwords)
