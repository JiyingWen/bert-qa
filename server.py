#!/usr/bin/env python3
import os,  sys
try:
	my_path = os.path.dirname(os.path.realpath(__file__))
except:
	my_path = os.path.normpath(join(os.getcwd(), path))
import code
import logging
from retriever.drqa import retriever
from retriever.drqa.retriever import utils
import sqlite3
import requests
import json
from flask import Flask, request, jsonify

# ------------------------------------------------------------------------------
# assets
# ------------------------------------------------------------------------------
doc_db = os.path.join(my_path, 'retriever/doc_db/sap_wiki_page.db')
retriever_model = os.path.join(my_path, 'retriever/tfidf_model/sap_wiki_page-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz')
bert_url = 'http://localhost:5000/predict'

# ------------------------------------------------------------------------------
# utils
# ------------------------------------------------------------------------------
def fetch_doc_text(doc_id, conn):
	"""
	fetch document  by doc_id
	"""
	cursor = conn.cursor()
	cursor.execute(
		"SELECT text FROM documents WHERE id = ?",
		(utils.normalize(doc_id),)
		)
	result = cursor.fetchone()
	cursor.close()
	return result if result is None else result[0]

def process(query, conn, k=1):
	"""
	retrieve the top k document based on query
	"""
	ranker = retriever.get_class('tfidf')(retriever_model)
	doc_names, doc_scores = ranker.closest_docs(query, k)
	context = fetch_doc_text(doc_names[0], conn)
	return context

def send_question_context_to_bert(query, context, qid):
	"""
	send question/context pairs to bert to identify answer span
	"""
	headers = {
		'Content-Type':'application/json',
	}
	payload = {
		'options':{
			'n_best':'true',
			'n_best_size':5,
			'max_answer_length':30
		},
		'data':[
		{
			'id':qid,
			'question':query,
			'context':context
		}
		]
	}

	response = requests.post(bert_url, headers = headers, data = json.dumps(payload))
	response.raise_for_status()
	result = response.json()
	return result

# ------------------------------------------------------------------------------
# flask app
# ------------------------------------------------------------------------------

app = Flask(__name__)

@app.route('/get-answer', methods=['POST'])
def get_answer():
	# get question from chat history
	data = json.loads(request.get_data().decode('utf-8'))
	question = data['nlp']['source']
	qid = data['nlp']['uuid']
	app.logger.info("Trying to get answer for: " + question)
	app.logger.info('Initializing ranker...')
	# connect to database
	conn = sqlite3.connect(doc_db)
	# get paragraph/context that contains the answer
	context = process(question, conn, k = 1)
	result = send_question_context_to_bert(question, context, qid)
	return jsonify(
		status = 200, 
		replies = [{
		'type' : 'text',
		'content': result['response']['result'][0]['best_prediction']
		}])


# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------

if __name__ == '__main__':
	app.run(port=5005, host = '0.0.0.0')

