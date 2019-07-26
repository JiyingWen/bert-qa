import os
import tensorflow as tf
import modeling
from run_squad import model_fn_builder

try:
	my_path = os.path.dirname(os.path.realpath(__file__))
except:
	my_path = os.path.normpath(join(os.getcwd(), path))

MODEL_ROOT = os.path.join(my_path, 'uncased_L-12_H-768_A-12') ## bert base files
MODEL_FT = os.path.join(my_path, 'bert_output/squad1_1_base') ## fine-tuned squad model
INIT_CKPT = 'model.ckpt-29199'
EXPORT_PATH = 'bert_squad_pb_export'

batch_size = 1
max_seq_length = 384  ## same on trained model (for question + content)

def serving_input_receiver_fn():
	feature_spec = {
		"unique_ids": tf.FixedLenFeature([], tf.int64),
		"input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
		"input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
		"segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
	}
	"""An input receiver that expects a serialized tf.Example."""
	serialized_tf_example = tf.placeholder(dtype=tf.string,
										 shape=[batch_size],
										 name='input_example_tensor')
	receiver_tensors = {'examples': serialized_tf_example}
	features = tf.parse_example(serialized_tf_example, feature_spec)
	return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

	
bert_config_file = os.path.join(MODEL_ROOT, 'bert_config.json')
init_ckpt_file = os.path.join(MODEL_FT, INIT_CKPT)

bert_config = modeling.BertConfig.from_json_file(bert_config_file)

is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

run_config = tf.contrib.tpu.RunConfig(
		cluster=None,
		master=None,
		model_dir=MODEL_FT,
		save_checkpoints_steps=5000,
		tpu_config=tf.contrib.tpu.TPUConfig(
			iterations_per_loop=1000,
			num_shards=8,
			per_host_input_for_training=is_per_host))

model_fn = model_fn_builder(
		bert_config=bert_config,
		init_checkpoint=init_ckpt_file,
		learning_rate=None,
		num_train_steps=None,
		num_warmup_steps=None,
		use_tpu=False,
		use_one_hot_embeddings=False)

estimator = tf.contrib.tpu.TPUEstimator(
		use_tpu=False,
		model_fn=model_fn,
		config=run_config,
		train_batch_size=6,
		predict_batch_size=8)

estimator._export_to_tpu = False  ## !!important to add this

estimator.export_saved_model(export_dir_base = EXPORT_PATH, serving_input_receiver_fn = serving_input_receiver_fn)

