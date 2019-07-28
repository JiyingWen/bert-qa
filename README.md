# Knowledge base Question Answering Chatbot

![Alt text](misc/chatbot.png?raw=true "Chatbot UI")

We demonstrate an end-to-end question answering system that integrates BERT with an information retriever system. The architecture, as shown below,  is comprised of two main modules: the document retriever and the BERT reader. The retriever is responsible for selecting paragraphs of text that contain the answer, which is then passed to the reader to identify an answer span.

![Alt text](misc/architecture.png?raw=true "System Architecture")

For the retriever system, we used Document Retriever of DrQA, which is a traditional information retrieval system which use TF-IDF rankings; for the reader model, we used a BERT-based reader fine-tuned with SQuAD 1.1.

The interface of the QA chatbot is using [SAP Conversational AI](https://cai.tools.sap/), which is a bot building platform that allows businesses to construct natural dialogue services easily and quickly. The main part, `server.py`, is wrapped into a flask app with endpoint `/get-answer`, which collects question text that user typed in, retrieves the paragraph that most likely to contain the answer via Document Retriever, calls the BERT Reader to get the answer, and sends the answer back to the chat window. 
