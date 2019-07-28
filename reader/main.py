
from __future__ import print_function
import sys, os, time
import json, collections
import numpy as np
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from flask import Flask, request, jsonify
import tokenization
from run_squad import *

try: 
	my_path = os.path.dirname(os.path.realpath(__file__))
except:
	my_path = os.path.normpath(join(os.getcwd(), path))

os.chdir(my_path)

# ------------------------------------------------------------------------------
# configs
# ------------------------------------------------------------------------------
batch_size = 1  ## new batch size
# model configs match training
max_seq_length = 384
doc_stride = 128 ## same as triained model
max_query_length = 64 ## len limit for question
# grpc port
grpc_tfserv = '127.0.0.1:8500'

# ------------------------------------------------------------------------------
# assets
# ------------------------------------------------------------------------------
vocab_file = os.path.join(my_path, 'uncased_L-12_H-768_A-12/vocab.txt')
tokenizer = tokenization.FullTokenizer(
			vocab_file=vocab_file, do_lower_case=True)
# feature writing to disk before request 
output_dir = 'temp'
tf.gfile.MakeDirs(output_dir)
predict_file = os.path.join(output_dir, "new.tf_record")

# ------------------------------------------------------------------------------
# utils
# ------------------------------------------------------------------------------
def read_data(input_data):
	def is_whitespace(c):
		if c == ' ' or c == '\t' or c == '\r' or c == '\n' or ord(c) == 0x202F:
			return True
		return False

	examples = []

	for inputs in input_data:
		qas_id = inputs['id']
		paragraph_text = inputs['context']
		question_text = inputs['question']
		doc_tokens = []

		prev_is_whitespace = True

		for c in paragraph_text:
			if is_whitespace(c):
				prev_is_whitespace = True
			else:
				if prev_is_whitespace:
					doc_tokens.append(c)
				else:
					doc_tokens[-1] += c
				prev_is_whitespace = False

		example = SquadExample(
			qas_id = qas_id,
			question_text = question_text,
			doc_tokens = doc_tokens,
			orig_answer_text = "",
			start_position = -1,
			end_position = -1,
			is_impossible = False
			)
		examples.append(example)
	return examples

def process_inputs(input_data):
	eval_examples = read_data(input_data)
	eval_features = []
	
	eval_writer = FeatureWriter(
			filename=predict_file,
			is_training=False)

	def append_feature(feature):
		eval_features.append(feature)
		eval_writer.process_feature(feature)

	convert_examples_to_features(
			examples=eval_examples,
			tokenizer=tokenizer,
			max_seq_length=max_seq_length,
			doc_stride=doc_stride,
			max_query_length=max_query_length,
			is_training=False,
			output_fn=append_feature)
	eval_writer.close()

	return eval_examples, eval_features

def process_result(result):
	
	unique_id = int(result["unique_ids"].int64_val[0])
	start_logits = [float(x) for x in result["start_logits"].float_val]
	end_logits = [float(x) for x in result["end_logits"].float_val]

	formatted_result = RawResult(
			unique_id = unique_id,
			start_logits = start_logits,
			end_logits = end_logits)
	
	return formatted_result

def process_output(all_results, eval_examples, eval_features, input_data, n_best, n_best_size, max_answer_length):
	all_predictions, all_nbest_json = write_predictions( eval_examples, 
														 eval_features, 
														 all_results,
														 n_best_size = n_best_size, 
														 max_answer_length = max_answer_length,
														 do_lower_case = True,
														 is_client = True)
														 # version_2_with_negative = False)	
	res = []
	for i in range(len(all_predictions)):
		id_ = input_data[i]["id"]
		if n_best:
			res.append(collections.OrderedDict({
					"id": id_,
					"question": input_data[i]["question"],
					"best_prediction": all_predictions[id_],
					"n_best_predictions": all_nbest_json[id_]
					}))
		else:
			res.append(collections.OrderedDict({
					"id": id_,
					"question": input_data[i]["question"],
					"best_prediction": all_predictions[id_]
					}))
	return res

def grpc_request(hostport, input_data, n_best, n_best_size, max_answer_length):

	channel = grpc.insecure_channel(hostport)
	stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
	print("** " + "Processing inputs...")
	eval_examples, eval_features = process_inputs(input_data)
	record_iterator = tf.python_io.tf_record_iterator(path=predict_file)
	all_results = []
	print("** " + "Calling model server...")
	for string_record in record_iterator:
		request = predict_pb2.PredictRequest()
		request.model_spec.name = 'bert_squad_v1' # match model name specified in target location docker run
		
		request.inputs['examples'].CopyFrom(
			tf.contrib.util.make_tensor_proto(string_record,
    										 dtype=tf.string,
   											 shape=[batch_size])
			)

		result_future = stub.Predict.future(request, 5.0)  # 5 seconds for timeout
		result = result_future.result().outputs
	
		all_results.append(process_result(result))
	print("** " + "Processing outputs...")
	res = process_output(all_results, eval_examples, eval_features, input_data, 
						n_best, n_best_size, max_answer_length)
	
	return res

# ------------------------------------------------------------------------------
# flask app
# ------------------------------------------------------------------------------
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
	t0 = time.time()
	try:
		json_input = json.loads(request.get_data())
		input_data = json_input["data"]
		if "options" in json_input:
			options = json_input["options"]
			n_best = options["n_best"]
			n_best_size = options["n_best_size"]
			max_answer_length = options["max_answer_length"]
		else:
			n_best = True
			n_best_size = 5
			max_answer_length = 30
	except Exception as err:
		return jsonify(status = 500,
				  		response = {"status": "FAILED",
				  					"error": "Input data format error, " + repr(err)
				  					}
				  		)
	try:
		res = grpc_request(grpc_tfserv, input_data, n_best, n_best_size, max_answer_length)
		return jsonify(status = 200,
						response = 
					  		{"latency": time.time() - t0,
					  		"result": res,
							"status": "SUCCESS"}
							)
	except Exception as err:
		return jsonify(status = 500,
				  		response = {"status": "FAILED",
				  					"error": "gRPC server error, " + repr(err)
				  					}
				  		)


if __name__ == '__main__':
	app.run(port = 5000)



