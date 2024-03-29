python run_squad.py \
  --do_train=true \
  --train_file=SQuAD/train-v1.1.json \
  --do_predict=true \
  --predict_file=SQuAD/dev-v1.1.json \
  --vocab_file=gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12/vocab.txt\
  --bert_config_file=gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint= gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=384 \
  --train_batch_size=6 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --doc_stride=128 \
  --save_checkpoints_steps=5000 \
  --output_dir=bert_output/squad_base
