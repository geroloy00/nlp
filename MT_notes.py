# =============================================
# 🧪 Midterm Exam: Airline Sentiment Analysis
# =============================================

# This script walks through a complete sentiment analysis pipeline
# using US airline tweets. Includes preprocessing, EDA, modeling (optional LDA), and evaluation.



# --- Code Block 1 ---
# (Insert explanation for this block based on its role)
# Install the pyLDAvis library
!pip install pyLDAvis


# --- Code Block 2 ---
# (Insert explanation for this block based on its role)
#importing required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
import warnings
import string
import spacy
import gensim

import pyLDAvis.gensim
import pyLDAvis.gensim_models as gensimvis

from sklearn.model_selection import StratifiedKFold, train_test_split
from mlxtend.plotting import plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from gensim import corpora, models
from gensim.models import Word2Vec
from gensim.similarities import MatrixSimilarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from collections import Counter
from gensim.matutils import corpus2csc, sparse2full, corpus2dense
from wordcloud import WordCloud
from sklearn.utils import resample
from wordcloud import WordCloud,STOPWORDS
from gensim.models.coherencemodel import CoherenceModel

%matplotlib inline

nltk.download('stopwords')
stop_words = stopwords.words('english')
nlp = spacy.load('en_core_web_sm')

# Visualize the topics
pyLDAvis.enable_notebook()
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# --- Code Block 3 ---
# (Insert explanation for this block based on its role)
# 1. Load the dataset from https://raw.githubusercontent.com/satyajeetkrjha/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv
# using as normal the pandas .read_csv() function. Store in a new variable named 'data'
# 2. Print the dimensionality of data
# 3. Preview the first few rows of data

data = pd.read_csv('https://raw.githubusercontent.com/satyajeetkrjha/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv')
print(data.shape)
data


# --- Code Block 4 ---
# (Insert explanation for this block based on its role)
# Get the overall information of the data DataFrame (data types, missing values, etc.)

data.info()


# --- Code Block 5 ---
# (Insert explanation for this block based on its role)
# Get all the unique airline names within 'data' and store them in a variable 'airlines'. Which feature would you use?
# This list qill be used later on in the analysis
# Print the 'airlines' results

print('Unique Airline Names:')
airlines = data['airline'].unique().tolist()
airlines


# --- Code Block 6 ---
# (Insert explanation for this block based on its role)
# 1. Set the column 'tweet_id' as the index of data (if you don't know how, you can drop it. Alternatively, the next step will sort this out).
# Remember to conduct this step inplace or with assignment back to "data" for any changes to take effect
# 2. Select and keep ONLY a subset of the features/columns: keep only 'text' and 'airline_sentiment' columns, and re-assign back to data (overwrite data)
# 3. Preview once more the results of data as a sanity check. Have your changes gone through?

data.set_index('tweet_id', inplace = True)
data


# --- Code Block 7 ---
# (Insert explanation for this block based on its role)
data = data[['text', 'airline_sentiment']]
data


# --- Code Block 8 ---
# (Insert explanation for this block based on its role)
# Drop the duplicates within data. Remember to assign back or conduct this step with replacement.
# Sanity check: print the dimensionality of 'data' before and after the drop. Was your drop successful?

data.shape


# --- Code Block 9 ---
# (Insert explanation for this block based on its role)
data = data.drop_duplicates()
data.shape


# --- Code Block 10 ---
# (Insert explanation for this block based on its role)
# Check for null values in data; you can return True/False, counts or any other solution of your choice to investigate for missing values

data.isnull().sum().sum() 


# --- Code Block 11 ---
# (Insert explanation for this block based on its role)
# Can you print the 'text' (column) of one random sample in data?

data['text'].sample(1)


# --- Code Block 12 ---
# (Insert explanation for this block based on its role)
# Get the number (frequency/count) of instances (samples) per each sentiment class (this will be a useful insight for the ML classifier later on).
# Store in a variable with a name of your choice and print the results

countpersentiment = data['airline_sentiment'].value_counts()
countpersentiment


# --- Code Block 13 ---
# (Insert explanation for this block based on its role)
# Use the variable above to plot a seaborn countplot

sns.countplot(x="airline_sentiment", data=data);


# --- Code Block 14 ---
# (Insert explanation for this block based on its role)
# - What do you observe?? Briefly describe here your findings
# we can easily observe that the majority of the comments that the airline receives are by far negative ones (around 9k), 
# followed by neutral ones (around 3k) and then followed by positive ones (around 2k)


# --- Code Block 15 ---
# (Insert explanation for this block based on its role)
#  Bonus activity: can you filter and draw a Wordcloud of only the positive sentiment cases? Use also STOPWORDS from WordCloud

df_pos = data[data['airline_sentiment'] == "positive"]

wordcloud = WordCloud(stopwords = STOPWORDS,max_words = 1000 , width = 1600 , height = 800,
                     collocations=True).generate(" ".join(df_pos['final_text']))
plt.imshow(wordcloud)


# --- Code Block 16 ---
# (Insert explanation for this block based on its role)
#  Bonus activity: can you filter and draw a Wordcloud of only the negative sentiment cases? Use also STOPWORDS from WordCloud

df_neg = data[data['airline_sentiment'] == "negative"]

wordcloud = WordCloud(stopwords = STOPWORDS,max_words = 1000 , width = 1600 , height = 800,
                     collocations=True).generate(" ".join(df_neg['final_text']))
plt.imshow(wordcloud)



# --- Code Block 17 ---
# (Insert explanation for this block based on its role)
# Define a function named 'lowercase' to lowercase your text in data

def lowercase(text):
    return text.lower() 


# --- Code Block 18 ---
# (Insert explanation for this block based on its role)
# Apply your 'lowercase' function to lowercase your data on the 'text' feature of data.
# Remember to reassign back to the same column for the changes to take effect.
# Preview your results

data['text'] = data['text'].apply(lowercase)
data['text'].head()


# --- Code Block 19 ---
# (Insert explanation for this block based on its role)
# Define a function named 'remove_stopwords' to remove stopwords from your text.
# BONUS / Extra points: you may also wish to append to your stopwords the names of the airlines that you had previously stored in the variable 'airlines'
# You may also consider pre-processing these values before appending them.
# Optionally, you can also append any other frequently encountered words in your stopwords.

def remove_stopwords(text):
    textArr = text.split(' ')
    rem_text = " ".join([i for i in textArr if i not in stop_words])
    return rem_text


# --- Code Block 20 ---
# (Insert explanation for this block based on its role)
# Apply your 'remove_stopwords' function to remove the stopwords from your 'text' feature of data.
# Remember to reassign back to the same column for the changes to take effect.
# Preview your results

data['text'] = data['text'].apply(remove_stopwords)
data['text'].head()


# --- Code Block 21 ---
# (Insert explanation for this block based on its role)
# Define one or more functions to perform ****at least one more additional**** pre-processing step of your choice based on the data that you observe!
# The cleaner the data, the higher the mark for this activity :D

def remove_usernames(text):
    if isinstance(text, str):
        text = re.sub(r'@[^\s]+', '', text)
    return text


# --- Code Block 22 ---
# (Insert explanation for this block based on its role)
# Justify and briefly describe your choice of pre-processing step(s) here

data['text'] = data['text'].apply(remove_usernames)
data['text'].head()

#i chose to remove the usernames in the comments because i didn't want to keep the mentioned names


# --- Code Block 23 ---
# (Insert explanation for this block based on its role)
# Create a list of the 'text' column from data. Store in a new variable named 'text_list'.
# Preview the first entry of your text_list

text_list = data['text'].values.tolist()
text_list[0]


# --- Code Block 24 ---
# (Insert explanation for this block based on its role)
type(text_list)


# --- Code Block 25 ---
# (Insert explanation for this block based on its role)
# Define a function to perform lemmatization, named 'lemmatization'.
# Bonus: allow postags only NOUN and ADJ

nlp = spacy.load('en_core_web_sm')
def lemmatization(text, allowed_postags=['NOUN', 'ADJ']): 
       output = []
       for sent in text:
            doc = nlp(sent) 
            output.append([token.lemma_ for token in doc if token.pos_ in allowed_postags ])
       return output


# --- Code Block 26 ---
# (Insert explanation for this block based on its role)
# Run your lemmatization function on text_list (remember, this is a list, not a DataFrame so you cannot use apply())
# Store in a new variable named 'tokenized_reviews'
### THIS STEP MAY TAKE A WHILE TO EXECUTE ###

tokenized_reviews = lemmatization(text_list)
tokenized_reviews


# --- Code Block 27 ---
# (Insert explanation for this block based on its role)
# Concatenate the tokens into single sentences once again, creating a new feature within data named 'final_text'

data['final_text']= ''

for i in range(len(text_list)):
    data['final_text'].iloc[i] = "".join(text_list[i])
data.head()


# --- Code Block 28 ---
# (Insert explanation for this block based on its role)
# Store the final_text column from data as your variable X
# Store the airline_sentiment column from data as your class vector y

X = data['final_text']
y = data['airline_sentiment']
X
y


# --- Code Block 29 ---
# (Insert explanation for this block based on its role)
# In this step we want to map the classes 'positive', 'neutral' and 'negative' to numerical values
# Instantiate a LabelEncoder() object and store in a variable named 'le'
# Fit and transform your LabelEncoder to your y class variable

le = LabelEncoder()
y = le.fit_transform(y)
y


# --- Code Block 30 ---
# (Insert explanation for this block based on its role)
# Use the holdout method from sklearn to split your data into train and test. Set the random_state to 42 and use stratification.
# Print the dimensionality of the end results as a sanity check

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, stratify = y)
print('X_train shape: ', X_train.shape)
print('X_test shape: ', X_test.shape)


# --- Code Block 31 ---
# (Insert explanation for this block based on its role)
# Import the library of the vectorization technique of your choice

from sklearn.feature_extraction.text import CountVectorizer


# --- Code Block 32 ---
# (Insert explanation for this block based on its role)
# Briefly justify your choice for this particular vectorizer

# i chose the “bags-of-words” representation which ignores structure and simply counts how often each word occurs. 
# CountVectorizer allows us to use the bags-of-words approach, by converting a collection of text documents into
# a matrix of token counts.


# --- Code Block 33 ---
# (Insert explanation for this block based on its role)
# Instantiate a vectorizer of your choice into a new variable named 'vect'

vect = CountVectorizer().fit(X_train)
vect


# --- Code Block 34 ---
# (Insert explanation for this block based on its role)
# Can you also create/ instantiate your vectorizer using unigrams and bigrams?
# Optional arguments to consider can be: min_df, max_df and max_features
# Store in a variable 'vect_grams'

vect_grams = CountVectorizer(min_df = 5, max_df = 20, ngram_range = (1,2)).fit(X_train)


# --- Code Block 35 ---
# (Insert explanation for this block based on its role)
# Apply any of the two afore-mentioned vectorizers to your train and test data to convert the text into numbers,
# and store the results in X_train_vec and X_test_vec respectively

X_train_vec = vect.transform(X_train)
X_test_vec = vect.transform(X_test)


# --- Code Block 36 ---
# (Insert explanation for this block based on its role)
# Import the Supervised Learning algorithm (classification model) of your choice

from sklearn.svm import LinearSVC


# --- Code Block 37 ---
# (Insert explanation for this block based on its role)
# 1. Instantiate your ML model using any hyperparameters of your choice (no need for tuning)
# 2. Fit your model to your train data
# 3. Predict your test data. Store into a variable named 'pred'
# 4. Report the accuracy score between your predicted and real test values

SVCmodel = LinearSVC()
SVCmodel.fit(X_train_vec, y_train)
y_pred = SVCmodel.predict(X_test_vec)


# --- Code Block 38 ---
# (Insert explanation for this block based on its role)
# Print the classification report

print(classification_report(y_test, y_pred))


# --- Code Block 39 ---
# (Insert explanation for this block based on its role)
# Build the confusion matrix between your predicted and real test values
# Store into a new variable named 'cm'

cm = confusion_matrix(y_test, y_pred)
cm


# --- Code Block 40 ---
# (Insert explanation for this block based on its role)
# Extra: plot in a heatmap the confusion matrix

f, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(cm, 
            annot=True, 
            annot_kws={'size': 8}, 
            cmap="Spectral_r");


# --- Code Block 41 ---
# (Insert explanation for this block based on its role)
# What do you observe in your results? Describe your findings

# we can see that the true mapped values to 0 and the the predicted mapped values to 0 have the highest correlation


# --- Code Block 42 ---
# (Insert explanation for this block based on its role)
# Create a corpora.Dictionary by passing the tokenized_reviews. Store in a new variable 'dictionary'

dictionary = corpora.Dictionary(tokenized_reviews)

texts = tokenized_reviews


# --- Code Block 43 ---
# (Insert explanation for this block based on its role)
print(tokenized_reviews[1])


# --- Code Block 44 ---
# (Insert explanation for this block based on its role)
# Create the doc_term_matrix from your tokenized_reviews

doc_term_matrix = [dictionary.doc2bow(rev) for rev in tokenized_reviews]


# --- Code Block 45 ---
# (Insert explanation for this block based on its role)
# Instantiate the LDA model and store it in a variable 'LDA''
# Build your LDA model using any parameters of your choice vy advising the documentation

# *********** Write comments next to every argument you have selected to use, justifying their entries/values

LDA = gensim.models.ldamodel.LdaModel

# Build LDA model
lda_model = LDA(corpus=doc_term_matrix,
                id2word=dictionary,
                num_topics=10, #set to 10
                random_state=100, #alpha and beta are hyperparameters that affect sparsity of the topics
                chunksize=100,
                passes=10, #controls how often we train the model on the entire corpus
                iterations=10) #controls how often we repeat a particular loop over each document


# --- Code Block 46 ---
# (Insert explanation for this block based on its role)
# Print the Keywords for each of the number of topics generated by the lda_model

lda_model.print_topics()


# --- Code Block 47 ---
# (Insert explanation for this block based on its role)
# Create a visualization plot of your LDA

vis = gensimvis.prepare(lda_model, doc_term_matrix, dictionary)
vis


# --- Code Block 48 ---
# (Insert explanation for this block based on its role)
### Explain your findings from LDA and the plot above

# we can see that the most frequent words in our dataset are: flight, customer, thank, hour, service, time
# for topic 1 the top 3 frequent words in our dataset are: customer, thank, service
# for topic 2 the top 3 frequent words in our dataset are: plane, flight, crew
# and so on


# --- Code Block 49 ---
# (Insert explanation for this block based on its role)
# Calculate the coherence of your lda_model

from gensim.models import CoherenceModel

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenized_reviews, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score: ', coherence_lda)


# --- Code Block 50 ---
# (Insert explanation for this block based on its role)
# Extra: calculate the perplexity of your lda_model

print('\nPerplexity: ', lda_model.log_perplexity(doc_term_matrix)) 


# --- Code Block 51 ---
# (Insert explanation for this block based on its role)
# Extra: Build your own compute_coherence_values() function to loop through various LDA model hyperparameters to find the optimal LDA model


# supporting function
def compute_coherence_values(corpus, dictionary, k, a, b):
    
    lda_model = gensim.models.LdaMulticore(corpus=doc_term_matrix,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts= tokenized_reviews, dictionary=dictionary, coherence='c_v')
    
    return coherence_model_lda.get_coherence()



# --- Code Block 52 ---
# (Insert explanation for this block based on its role)
# Extra: Execute the function defined above; this may take a while to run   

#stopped running it due to time limitations

import numpy as np
import tqdm

grid = {}
grid['Validation_Set'] = {}

# Topics range
min_topics = 2
max_topics = 11
step_size = 1
topics_range = range(min_topics, max_topics, step_size)

# Alpha parameter
alpha = list(np.arange(0.01, 1, 0.3))
alpha.append('symmetric')
alpha.append('asymmetric')

# Beta parameter
beta = list(np.arange(0.01, 1, 0.3))
beta.append('symmetric')

# Validation sets
num_of_docs = len(doc_term_matrix)
corpus_sets = [gensim.utils.ClippedCorpus(doc_term_matrix, int(num_of_docs*0.75)), 
               doc_term_matrix]

corpus_title = ['75% Corpus', '100% Corpus']

model_results = {'Validation_Set': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []
                }

# Can take a long time to run
if 1 == 1:
    pbar = tqdm.tqdm(total=(len(beta)*len(alpha)*len(topics_range)*len(corpus_title)))
    
    # iterate through validation corpuses
    for i in range(len(corpus_sets)):
        # iterate through number of topics
        for k in topics_range:
            # iterate through alpha values
            for a in alpha:
                # iterare through beta values
                for b in beta:
                    # get the coherence score for the given parameters
                    cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=dictionary, 
                                                  k=k, a=a, b=b)
                    # Save the model results
                    model_results['Validation_Set'].append(corpus_title[i])
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Beta'].append(b)
                    model_results['Coherence'].append(cv)
                    
                    pbar.update(1)
    pd.DataFrame(model_results).to_csv('./results/lda_tuning_results.csv', index=False)
    pbar.close()


# --- Code Block 53 ---
# (Insert explanation for this block based on its role)
#stopped running it due to time limitations
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# --- Code Block 54 ---
# (Insert explanation for this block based on its role)
#stopped running it due to time limitations
model_list, coherence_values = compute_coherence_values(dictionary=dictionary,
                                                        corpus=doc_term_matrix,
                                                        texts=tokenized_reviews,
                                                        start=2,
                                                        limit=50,
                                                        step=1)


# --- Code Block 55 ---
# (Insert explanation for this block based on its role)
# Extra: plot the coherence values

#stopped running it due to time limitations

limit=50; start=2; step=1;
x = range(start, limit, step)

plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show() # Print the coherence scores



# --- Code Block 56 ---
# (Insert explanation for this block based on its role)
# Extra: Loop through the number of topics and coherence values

#stopped running it due to time limitations

for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# --- Code Block 57 ---
# (Insert explanation for this block based on its role)
# Extra: Find the optimal number of topics - Justify why



# --- Code Block 58 ---
# (Insert explanation for this block based on its role)
# Extra: Build the optimal model and plot again the results.




# --- Code Block 59 ---
# (Insert explanation for this block based on its role)
# Extra: What do you observe? Discuss briefly any final findings.




# --- Code Block 60 ---
# (Insert explanation for this block based on its role)



# --- Code Block 61 ---
# (Insert explanation for this block based on its role)
# Extra: Import the Word2Vec library

import gensim
from gensim.models import Word2Vec


# --- Code Block 62 ---
# (Insert explanation for this block based on its role)
# Extra: Instantiate and create a  Word2Vec() model using the tokenized_reviews.
# Optionally, set other parameters like the vector_size = 100,  window, min_count, and sg (for CBOW or skip-gram)
# Store in a variable named 'w2v'

w2v = Word2Vec(sentences=tokenized_reviews, vector_size=100, min_count=1, epochs=100) 

print(w2v)


# --- Code Block 63 ---
# (Insert explanation for this block based on its role)
# Extra: using w2v, find the most similar words to the word 'crew'

w2v.wv.most_similar('crew')


# --- Code Block 64 ---
# (Insert explanation for this block based on its role)
# Extra: using w2v, find the most similar words to the word 'delay'

w2v.wv.most_similar('delay')


# --- Code Block 65 ---
# (Insert explanation for this block based on its role)

