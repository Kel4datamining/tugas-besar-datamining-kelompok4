from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from gensim.models import Word2Vec
from sklearn.pipeline import Pipeline

def train_naive_bayes(X, y, method="tfidf", use_chi2=False, k_best=1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if method == "tfidf":
        vectorizer = TfidfVectorizer()
    elif method == "bow":
        vectorizer = CountVectorizer()
    elif method == "word2vec":
        return train_w2v(X, y)
    else:
        raise ValueError("Metode tidak dikenali")

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    if use_chi2:
        selector = SelectKBest(chi2, k=min(k_best, X_train_vec.shape[1]))
        X_train_vec = selector.fit_transform(X_train_vec, y_train)
        X_test_vec = selector.transform(X_test_vec)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    return model, y_test, y_pred

def train_w2v(X, y):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import train_test_split
    import numpy as np

    tokenized = [text.split() for text in X]
    w2v_model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4)
    
    def document_vector(doc):
        return np.mean([w2v_model.wv[word] for word in doc if word in w2v_model.wv], axis=0)

    vectors = [document_vector(doc) for doc in tokenized]
    X_train, X_test, y_train, y_test = train_test_split(vectors, y, test_size=0.2, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_test, y_pred
