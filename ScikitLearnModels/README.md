The models saved here can be loaded using the joblib.load function from the sklearn.externals library and used like so:
```
from sklearn.externals import joblib

E_MLP = joblib.load('E_MLP.pkl')

predictions = E_MLP.predict(word_vectors)
```
Some notes on using these models:
* For all of the models, the first 1100 features for any particular observation are assumed to be: (1-300) GloVe trained on the Common Crawl, (301-500) GloVe trained on Twitter, (501-800) GloVe trained on Wikipedia, and (801-1100) Word2Vec trained on Google News. Just these features arranged in this way will satisfy the E_MLP model as well as identity, modifier, and behavior models.
* The P_MLP model uses the predictions of E_MLP as a feature, and A_MLP uses the predictions of both E_MLP and P_MLP as features, so you'll need to run them sequentially if you want to use them out-of-the-box. See the `pred` function in the analysis scripts for help in using these models.
