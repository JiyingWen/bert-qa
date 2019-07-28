# bert-qa

![Alt text](misc/architecture.png?raw=true "System Architecture")


We demonstrate an end-to-end question answering system that integrates BERT with an information retriever system. The architecture is comprised of two main modules: the document retriever and the BERT reader. The retriever is responsible for selecting paragraphs of text that contain the answer, which is then passed to the reader to identify an answer span.

For the retriever system, we used Document Retriever of DrQA, which is a traditional information retrieval system which use TF-IDF rankings; for the reader model, we used a BERT-based reader fine-tuned with SQuAD 1.1.

The UI for the QA chatbot is using [SAP Conversational AI](https://cai.tools.sap/). The main part is wrapped into a flask app (server.py) with endpoint `/get-answer`, which parses question text from chat json, gets the paragraph that might contain the answer via Document Retriever, and call the BERT Reader to get the answer. 
