In this directory you will find anlysis scripts in both a Jupyter notebook and a regular Python file. Both use Python3. Both should run without any errors, assuming the following conditions are true:

All files in this directory are downloaded and in the same directory (so they can access each other)
All the relevant libraries are downloaded onto your machine (see the first block of code)
All the relevant GloVe files are downloaded onto your machine in the current directory (see https://nlp.stanford.edu/projects/glove/). The Word2Vec library is loaded through the Gensim library, so you don't need to download that specifically.
You have sufficient memory available. The current approach in these scripts is to load all of the distributed representation libraries into memory, which is pretty memory intensive.
