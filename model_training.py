import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import joblib
from libml.preprocess import preprocess_text

# Load data
dataset = pd.read_csv('data/a1_RestaurantReviews_HistoricDump.tsv', delimiter='\t', quoting=3)
corpus = clean_review(dataset) 

# Feature extraction
cv = CountVectorizer(max_features=1420)
X = cv.fit_transform(corpus).toarray()
y = dataset['Liked'].values

# Save BoW
joblib.dump(cv, 'models/c1_BoW_v1.pkl')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train model
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Save model
joblib.dump(classifier, 'models/c2_Classifier_v1.pkl')