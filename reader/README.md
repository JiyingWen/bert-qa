# BERT Reader

BERT reader is based on [Google’s reference implementation](https://github.com/google-research/bert)(Tensorflow 1.13.0). For training, we begin with the BERT-Base model (uncased, 12-layer, 768-hidden, 12-heads, 110M parameters) and then fine-tune the model on training set of SQuAD (v1.1). All inputs to the reader are padded to 384 tokens; the learning rate is set to 3 * 10-5 and all other defaults settings are used. 

After training, model checkpoint is stored in the output directory. In order to serve the model, we need to export the `ckpt` file into SavedModel format that is used by Tensorflow Serving; this can be done by running `python export.py`. Then we serve the model using [Tensorflow Serving with Docker](https://www.tensorflow.org/tfx/serving/docker).

We wrap the BERT reader into a flask server, `main.py`, for making predictions in the inference time.The reason we need flask server is that when we make request to the TF server, we need to pass params in appropriate format. 
