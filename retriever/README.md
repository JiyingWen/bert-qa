# Document Retriever

The Document Retriever is the same retreiver of DrQA, which is a TF-IDF retrieval system built upon the knowledge base of your choice (e.g. here we used the wikipedia page of SAP as an example). The knowledge base is pre-segmented into paragraphs and indexed. The unit of retrieval is paragraph. The retriever measures the similarity between a given query and paragraphs by using dot product of TF-IDF weighted bag-of-words vectors. The vectors are computed after hashing bigrams to 2^24 bins with unsigned murmur3 hash, where murmur3 is a non-cryptographic hash function that hashes unigram-bigram tokens to bins and similar tokens hashed to the same bin. 

The retriever has two pre-requisites: storing the documents in a sqlite database, and building a TF-IDF weighted word-doc sparse matrix from the documents. For detailed steps, please refer to the [instructions](https://github.com/facebookresearch/DrQA/tree/master/scripts/retriever) on the orginal Github repo of DrQA.
