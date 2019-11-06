The models saved here can be loaded using the joblib.load function from the sklearn.externals library and used like so:
```
from sklearn.externals import joblib

E_MLP = joblib.load('E_MLP.pkl')

predictions = E_MLP.predict(word_vectors)
```
