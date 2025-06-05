# ===============================
# 📘 NLP Pipeline Cheatsheet
# ===============================

# --- 1. TEXT PREPROCESSING ---

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Lowercase
text = text.lower()

# Remove URLs
text = re.sub(r"http\S+|www\.\S+", "", text)

# Remove non-alphabetic characters
text = re.sub(r"[^a-zA-Z\s]", "", text)

# Tokenize
words = word_tokenize(text)

# Remove stopwords
stop_words = set(stopwords.words("english"))
words = [w for w in words if w not in stop_words and len(w) > 2]

# Lemmatization helper
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in words]

# --- 2. WORD EMBEDDINGS ---

from gensim.models import Word2Vec

# Prepare data
sentences = [word_tokenize(doc.lower()) for doc in docs]

# Train Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Access vectors
vector = model.wv['food']

# Similar words
model.wv.most_similar('food')

# --- 3. SUPERVISED LEARNING (CLASSIFICATION) ---

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Vectorize text
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(docs)
y = labels  # e.g., 0 = negative, 1 = neutral, 2 = positive

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# --- 4. TOPIC MODELING (LDA) ---

import gensim
from gensim import corpora
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# Tokenize
tokenized = [word_tokenize(doc.lower()) for doc in docs]

# Dictionary and corpus
id2word = corpora.Dictionary(tokenized)
corpus = [id2word.doc2bow(text) for text in tokenized]

# LDA model
lda_model = gensim.models.LdaModel(corpus, num_topics=5, id2word=id2word, passes=10)

# Display topics
for idx, topic in lda_model.print_topics():
    print(f"Topic {idx}: {topic}")

# Visualization
pyLDAvis.enable_notebook()
vis = gensimvis.prepare(lda_model, corpus, id2word)
pyLDAvis.display(vis)

# --- 5. EXAM CASE (Airline Sentiment) ---

# Common fields:
# - 'airline_sentiment': target class
# - 'text': input feature
# - Optional: 'negativereason' as secondary target

# Use same preprocessing + classification pipeline on tweet data
# You can also do topic modeling only on negative tweets to discover top complaint types
