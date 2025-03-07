import string
import pandas as pd # for handling csv data

import nltk
# import nltk modules for text processing
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#import scikit learn modules for converting text into numerical data to train ML model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# nltk.download('stopwords') # uncomment if stopwords needs to be downloaded

def preprocess_text(text, stemmer, stopwords_set):
    """
    preprocesses email text by converting the text to lowercase,
    removing punctuation, splitting into words, and applying
    stemming while removing stopwords

    :param text: the email text to process
    :param stemmer: instance of PorterStemmer for stemming words
    :param stopwords_set: a set of stopwords to remove from the text
    :return: the clean and preprocessed text
    """
    text = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    return ' '.join(text)

def train_spam_detector():
    """
    loads the dataset, preprocesses the text, and trains a RandomForest spam classifier.
    :return: trained classifier, vectorizer, stemmer, and stopwords set
    """
    # load dataset (CSV format with 'text' and 'label_num' columns)
    df = pd.read_csv('spam_ham_dataset.csv')

    # remove line breaks in text to ensure consistency
    df['text'] = df['text'].apply(lambda x: x.replace('\r\n', ' '))

    # initialize preprocessing tools
    stemmer = PorterStemmer()
    stopwords_set = set(stopwords.words('english'))

    # preprocess all email texts in the dataset
    corpus = [preprocess_text(text, stemmer, stopwords_set) for text in df['text']]

    # convert text into numerical vectors using CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus).toarray()
    y = df['label_num']

    # split the dataset into 80% training and 20% testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # create and train the RandomForest classifier
    clf = RandomForestClassifier(n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)

    # evaluate model performance based on test set
    accuracy = clf.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.4f}") # prints accuracy with four decimal places

    return clf, vectorizer, stemmer, stopwords_set

def classify_email(clf, vectorizer, stemmer, stopwords_set, email_text):
    """
    classifies if an email is spam or not.
    :param clf: the trained classifier
    :param vectorizer: the vectorizer used to transform email text
    :param stemmer: stemming tool used for text preprocessing
    :param stopwords_set: set of stopwords to remove from the email text
    :param email_text: the email content to classify as spam or not
    :return: "Spam if it is classified as so, otherwise not spam"
    """
    processed_email = preprocess_text(email_text, stemmer, stopwords_set)
    email_vector = vectorizer.transform([processed_email])
    prediction = clf.predict(email_vector)
    return "Spam" if prediction[0] == 1 else "Not Spam"

if __name__ == "__main__":
    # train spam detection model
    clf, vectorizer, stemmer, stopwords_set = train_spam_detector()

    # list of test emails
    test_emails = [
        "Congratulations! You have won a $1,000 gift card. Click here to claim now.",  # likely spam
        "Meeting at 10 AM tomorrow, please confirm your attendance.",  # not spam
        "URGENT: Your bank account is compromised. Verify your identity now!",  # potential spam
        "Limited-time offer! Get 90% off on all items. Click the link now!"  # potential spam
    ]

    # loop through and classify each email
    for email in test_emails:
        classification = classify_email(clf, vectorizer, stemmer, stopwords_set, email)
        print(f"Email: {email}")
        print(f"Classification: {classification}")
        print("-" * 70)