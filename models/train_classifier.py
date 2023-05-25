import sys
import re
import pandas as pd
import numpy as np
import pickle

from tqdm import tqdm
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')

from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,precision_recall_fscore_support




def load_data():
    # load data from database

    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql('Select * from InsertTableName', engine)
    print(df.columns)

    df = df[['message', 'related', 'request', 'offer',
             'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
             'security', 'military', 'water', 'food', 'shelter',
             'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
             'infrastructure_related', 'transport', 'buildings', 'electricity',
             'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
             'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
             'other_weather', 'direct_report']]
    df = df.dropna(axis=0)
    X = df['message']

    y = df[['related', 'request', 'offer',
            'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
            'security', 'military', 'water', 'food', 'shelter',
            'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
            'infrastructure_related', 'transport', 'buildings', 'electricity',
            'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
            'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
            'other_weather', 'direct_report']]
    categories = y.columns

    return X, y, categories


def tokenize(text):

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()

    stop_words = stopwords.words("english")

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if tok not in stop_words:
            clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LogisticRegression(class_weight="balanced",
                                                         random_state=88, C=0.1,
                                                         max_iter=100,
                                                         solver='liblinear')))

        ])
    # specify parameters for grid search
    """
    I have done a lot of experimentation and chosen the best option. 
    According to the task, I am using here Grid search only with two parameters, otherwise, the training will be too long.
    """
    parameters = {
        'clf__estimator__penalty': ['l1', 'l2'],  # type regularization (L1 or L2)
                }

    # Create grid search object
    model = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_macro')

    return model



def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    # Calculate and print classification report
    i = 0
    for column in Y_test:
        report = classification_report(list(Y_test[column]), y_pred[:, i])
        print(str(i + 1) + ') Colummn = ' + category_names[i])
        print(report)
        print("f1-score: ", precision_recall_fscore_support(list(Y_test[column]), y_pred[:, i], average='macro')[2])
        print("-----------------------------")
        i += 1




def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format('!!!')) #database_filepath
        X, Y, category_names = load_data() # database_filepath

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()