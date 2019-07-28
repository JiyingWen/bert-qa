# BERT Reader

BERT reader is based on Google’s reference implementation. For training, we begin with the BERT-Base model (uncased, 12-layer, 768-hidden, 12-heads, 110M parameters) and then fine-tune the model on training set of SQuAD (v2.0). All inputs to the reader are padded to 384 tokens; the learning rate is set to 3 * 10-5 and all other defaults settings are used. 
