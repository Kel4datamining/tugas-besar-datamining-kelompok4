from src.data_loader import load_data
from src.model import train_naive_bayes
from src.utils import evaluate_model, plot_conf_matrix

def main():
    print("Loading data...")
    df = load_data("data/processed/gojek_cleaned_labeled.csv")

    X = df['clean_text']
    y = df['sentiment']

    print("Training TF-IDF...")
    _, y_test, y_pred = train_naive_bayes(X, y, method="tfidf")
    evaluate_model(y_test, y_pred)
    plot_conf_matrix(y_test, y_pred, "TF-IDF")

    print("Training Bag-of-Words...")
    _, y_test, y_pred = train_naive_bayes(X, y, method="bow")
    evaluate_model(y_test, y_pred)
    plot_conf_matrix(y_test, y_pred, "BoW")

    print("Training Word2Vec...")
    _, y_test, y_pred = train_naive_bayes(X, y, method="word2vec")
    evaluate_model(y_test, y_pred)
    plot_conf_matrix(y_test, y_pred, "Word2Vec")

    print("Training TF-IDF + Chi2...")
    _, y_test, y_pred = train_naive_bayes(X, y, method="tfidf", use_chi2=True)
    evaluate_model(y_test, y_pred)
    plot_conf_matrix(y_test, y_pred, "TF-IDF + Chi2")

if __name__ == "__main__":
    main()
