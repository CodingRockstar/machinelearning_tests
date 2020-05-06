# Prediction of legal branches for legalcases

## Multi-Class text classification 4 beginners :D



### Prerequisites

* Python 3.7
* Tensorflow `pip install tensorflow`
* Numpy `pip install numpy`
* h5py `pip install h5py`
* Pandas `pip install pandas`
* nltk `pip install --user -U nltk`

**ATTENTION: You need a dataset of legal cases, which is by privacy reasons not included in this repo!**


##  1. Tests with Naive Bayes, SVM and logarithmic regression

```
python train_logreg.py
```


## 2. Tests with RNN (LSTM)

### Run it by:

1. create a trained model and save it with (for 1st run uncomment line 15 to download library with stopwords)

```
python train_lstm.py
```

2. get predictions on a validation dataset (file legalcase_validation.json is not included!) with

```
python predict_lstm.py
```


### PROBLEM

I had to limit the categories (legal branches) to 10 (from 78 in original dataset). With all categoiries I only got an accuracy by 50 % or less. In my opinion this is due to the poorly distributed data (about 50 % of the data relate to the 10 largest categories and the last 10 to 20 categories have just 1 or 2 occurances) and therefore I believe that I have way too little data. **If you can see a mistake in my assumptions or in my code, please do not hesitate to contact me.** :)