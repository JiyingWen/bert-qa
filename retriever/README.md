# Document Retriever

The retriever measures the similarity between a given query and articles by using dot product of TF-IDF weighted bag-of-words vectors. The vectors are computed after hashing bigrams to 2^24 bins with unsigned murmur3 hash, where murmur3 is a non-cryptographic hash function that hashes unigram-bigram tokens to bins and similar tokens hashed to the same bin. 

The retriever has two pre-requisites: storing the documents in a sqlite database, and building a TF-IDF weighted word-doc sparse matrix from the documents.
