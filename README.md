# bert-qa

![Alt text](misc/architecture.png?raw=true "Title")


We demonstrate an end-to-end question answering system that integrates BERT with an information retriever system. The architecture is comprised of two main modules: the document retriever and the BERT reader. The retriever is responsible for selecting paragraphs of text that contain the answer, which is then passed to the reader to identify an answer span.



For the retriever system, we used Document Retriever of DrQA, which is a traditional information retrieval system which use TF-IDF rankings; for the reader model, we used a BERT-based reader fine-tuned with SQuAD 1.1.

