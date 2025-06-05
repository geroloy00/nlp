# ========================================
# ✨ NLP Project Script (All Modules Merged)
# ========================================

# This script includes all steps from text preprocessing,
# embeddings, supervised learning, and topic modeling (LDA).
# It also includes guidance and comments based on your Midterm project.



# ===============================
# >>> SECTION: 02 PREPROCESSING
# ===============================
# Import Libraries
import kagglehub
import pandas as pd
# Download the dataset
# Download latest version
# https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
path = kagglehub.dataset_download("snap/amazon-fine-food-reviews")
print("Path to dataset files:", path)
# Load the dataset
df = pd.read_csv(path + "/Reviews.csv")
print(df.head())
# Print the information of the dataset
print(df.info())
# Extract Features and Target
# Score is the rating of the product. This will be our target variable.
df_score = df["Score"]
df_score
# Print the unique values of the target variable
df_score.unique()
# Print the statistics of the target variable
score_statistics = df_score.describe()
score_statistics
import matplotlib.pyplot as plt
# Plotting the bar chart for the value counts of the scores
score_counts = df_score.value_counts()
plt.figure(figsize=(10, 2))
score_counts.plot(kind='bar')
plt.title('Distribution of Scores')
plt.xlabel('Score')
plt.ylabel('Count')
plt.show()
# Summary is the title of the review. This will constitute our features.
df_summary = df["Summary"]
df_summary
# Importing numpy for generating random indices
import numpy as np
# Generating 10 random indices from the range of df_score length
rand_idxs = np.random.randint(0, len(df_score), size=10)
# Iterating over the random indices to print corresponding Score and Summary
for idx in rand_idxs:
    print(f"Score: {df_score.iloc[idx]} - Summary: {df_summary.iloc[idx]}")
# We zero out the data to free up memory
df = 0
# Preprocessing
The Preprocessing steps we will use are:
1. Lower Casing
2. Replacing URLs
3. Replacing Emojis
4. Replacing Usernames
5. Removing Non-Alphabets
6. Removing Consecutive letters
7. Removing Short Words
8. Removing Stopwords
9. Lemmatization
## Lowercase
def lowercase_text(text):
    # Convert text to lowercase.
    return str(text).lower()
# Apply lowercase function to all summaries
df_summary = df_summary.apply(lowercase_text)
# Display a few examples to verify the transformation
print("After lowercase transformation:")
rand_idxs = np.random.randint(0, len(df_summary), size=10)
for idx in rand_idxs:  
    print(f"Score: {df_score.iloc[idx]} - Summary: {df_summary.iloc[idx]}")
## Replace URLs
import re  # Importing the regular expressions module

# Define a regex pattern to identify URLs in the text
url_pattern = r"(?:https?|ftp)://[^\s/$.?#].[^\s]*"

def replace_urls(text):
    """
    Replace URLs in the text with the token 'URL'.
    Prints before and after if a replacement occurs.
    """
    text_str = str(text)
    replaced_text = re.sub(url_pattern, 'URL', text_str)

    if replaced_text != text_str:
        print(f"Before: {text_str}")
        print(f"After:  {replaced_text}\n")

    return replaced_text
# Apply URL replacement to all summaries
df_summary = df_summary.apply(replace_urls)
## Replacing Emojis
import re

# re.compile will compile the regex pattern into a regex object, necessary for 
# efficient pattern matching. This creates a reusable pattern object that can be
# used multiple times without recompiling the pattern each time, improving performance.
# u stands for Unicode
emoji_pattern = re.compile("["

    # Emoticons (e.g., 😀😁😂🤣😃😄😅😆)
    u"\U0001F600-\U0001F64F"  

    # Symbols & pictographs (e.g., 🔥🎉💡📦📱)
    u"\U0001F300-\U0001F5FF"  

    # Transport & map symbols (e.g., 🚗✈️🚀🚉)
    u"\U0001F680-\U0001F6FF"  

    # Flags (e.g., 🇺🇸🇬🇧🇨🇦 — these are pairs of regional indicators)
    u"\U0001F1E0-\U0001F1FF"  

    # Dingbats (e.g., ✂️✈️✉️⚽)
    u"\u2700-\u27BF"          

    # Supplemental Symbols & Pictographs (e.g., 🤖🥰🧠🦾)
    u"\U0001F900-\U0001F9FF"  

    # Symbols & Pictographs Extended-A (e.g., 🪄🪅🪨)
    u"\U0001FA70-\U0001FAFF"  

    # Miscellaneous symbols (e.g., ☀️☁️☂️⚡)
    u"\u2600-\u26FF"          

    "]+", flags=re.UNICODE)

# This pattern will match common text-based emoticons that aren't covered by the emoji Unicode ranges
# These emoticons are made up of regular ASCII characters like colons, parentheses, etc.
# Examples include:
# :) - happy face
# :( - sad face
# :D - laughing face
# ;) - winking face
emoticon_pattern = re.compile(r'(:\)|:\(|:D|:P|;\)|:-\)|:-D|:-P|:\'\(|:\||:\*)')
def remove_and_print(text):
    if emoji_pattern.search(text) or emoticon_pattern.search(text):
        print(f"Before: {text}")
        text = emoji_pattern.sub('', text)
        text = emoticon_pattern.sub('', text)
        print(f"After: {text}")
        print()
    return text
df_summary = df_summary.apply(remove_and_print)
## Replacing Usernames
import re

def replace_usernames(text):
    """
    Replace email addresses and true @usernames with 'USER'.
    Avoid matching embedded @ in profanity or stylized words.
    Print before and after if replacement occurs.
    """
    original = str(text)
    updated = original

    # Replace full email addresses
    updated = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', 'USER', updated)

    # Replace @usernames only when preceded by space, punctuation, or start of string
    updated = re.sub(r'(?:(?<=^)|(?<=[\s.,;!?]))@\w+\b', 'USER', updated)

    if updated != original:
        print(f"Before: {original}")
        print(f"After:  {updated}\n")
    
    return updated


# Apply username replacement to all summaries
df_summary = df_summary.apply(replace_usernames)
## Removing Non-Alphabets
import re

def clean_text(text, keep_punct=False):
    """
    Clean and normalize text for NLP classification tasks.
    
    Parameters:
    - text (str): The input text to be cleaned.
    - keep_punct (bool): 
        If True, retains key punctuation (. ! ?) which may carry emotional or contextual weight.
        If False, removes all non-alphabetic characters for simpler lexical analysis.
    
    Returns:
    - str: The cleaned text string, lowercased and stripped of unwanted characters.
    
    This function is designed for flexibility across different NLP tasks like sentiment analysis,
    topic classification, or spam detection. It handles:
    - Lowercasing text for normalization
    - Removing or preserving select punctuation
    - Removing digits, symbols, and special characters
    - Reducing multiple spaces to a single space
    - Optionally printing changes for debugging or logging

    When to use `keep_punct=True`:
    - Sentiment analysis: punctuation (e.g., "!", "?") can reflect strong emotion
    - Social media or informal text: expressive punctuation often carries signal
    - Sarcasm, emphasis, or tone-sensitive tasks

    When to use `keep_punct=False`:
    - Topic classification or document clustering: punctuation rarely adds value
    - Preprocessing for bag-of-words, TF-IDF, or topic modeling
    - When punctuation is inconsistent or noisy (e.g., OCR scans, scraped data)
    """
    
    # Convert input to string (safe handling)
    original = str(text)

    if keep_punct:
        # Keep only lowercase letters, spaces, and select punctuation (. ! ?)
        # Useful for capturing tone/sentiment
        cleaned = re.sub(r"[^a-z\s.!?]", "", original)
    else:
        # Keep only lowercase letters and spaces; remove all punctuation and symbols
        cleaned = re.sub(r"[^a-z\s]", "", original)

    # Normalize whitespace (collapse multiple spaces to one, strip leading/trailing)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # Optional: print before and after if a change occurred
    if original != cleaned:
        print(f"Before: {text}")
        print(f"After:  {cleaned}\n")

    return cleaned
# Apply non-alphabet removal to all summaries
df_summary = df_summary.apply(lambda x: clean_text(x, keep_punct=True))
## Removing Consecutive letters
def remove_consecutive_letters(text, max_repeat=2):
    """
    Normalize elongated words by limiting repeated characters.

    In informal or emotional text (e.g., reviews, tweets), users often repeat letters
    to add emphasis: "sooooo good", "loooove it", "greeaaat".
    
    This function reduces any character repeated more than `max_repeat` times 
    to exactly `max_repeat` occurrences (default: 2), preserving emphasis without bloating vocabulary.

    Parameters:
    - text (str): The input text
    - max_repeat (int): The maximum allowed repetitions for any character

    Returns:
    - str: Text with repeated characters normalized
    """
    text_str = str(text)
    pattern = r'(\w)\1{' + str(max_repeat) + r',}'
    cleaned = re.sub(pattern, r'\1' * max_repeat, text_str)

    # Print only if changes were made
    if cleaned != text_str:
        print(f"Before: {text_str}")
        print(f"After:  {cleaned}\n")

    return cleaned
# Apply consecutive letter removal to all summaries
df_summary = df_summary.apply(lambda x: remove_consecutive_letters(x, max_repeat=2))
## Removing Short Words
def remove_short_words(text, min_length=3, preserve_words=None):
    """
    Remove short words from text based on a minimum length threshold.
    
    Parameters:
    - text (str): The input text
    - min_length (int): Minimum word length to keep (default = 3)
    - preserve_words (set or list): Optional set of short but important words to keep (e.g., {'no', 'not'})
    
    Returns:
    - str: Text with short words removed, except for preserved ones
    
    Notes:
    - Use with care in sentiment analysis. Important short words like 'no', 'not', 'bad' may affect meaning.
    - Best used after stopword removal or on very noisy text.
    """
    preserve = set(preserve_words or [])
    words = str(text).split()
    filtered = [word for word in words if len(word) >= min_length or word.lower() in preserve]
    result = ' '.join(filtered)

    if result != text:
        print(f"Before: {text}")
        print(f"After:  {result}\n")

    return result
# Apply short word removal to all summaries
df_summary = df_summary.apply(lambda x: remove_short_words(x, min_length=3, preserve_words={'no', 'not'}))
## Removing Stopwords
# NLTK (Natural Language Toolkit) is a popular library for natural language processing in Python
# https://www.nltk.org/
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

print("Sample stopwords:", list(stopwords.words('english'))[:10])
# Define stopwords but keep critical ones like "not"
base_stopwords = set(stopwords.words('english'))
preserve = {'no', 'not', 'nor', 'never'}
custom_stopwords = base_stopwords - preserve
def remove_stopwords(text):
    """
    Remove stopwords from text, preserving key negation words.

    This function uses a customized stopword list that retains important
    short words like 'not', 'no', 'nor', and 'never' which carry significant
    meaning in tasks like sentiment analysis.

    Parameters:
    - text (str): Lowercased input text

    Returns:
    - str: Text with stopwords removed, but critical negation words preserved
    """
    words = str(text).split()
    filtered = [word for word in words if word not in custom_stopwords]
    result = ' '.join(filtered)

    if result != text:
        print(f"Before: {text}")
        print(f"After:  {result}\n")

    return result
# Apply remove_stopwords
df_summary = df_summary.apply(remove_stopwords)
## Lemmatization
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Download required NLTK resources
nltk.download('wordnet')  # Download WordNet, a lexical database of English words
nltk.download('omw-1.4')  # WordNet Lemmas sometimes need this, which is a mapping of WordNet lemmas to their Part of Speech (POS) tags.
nltk.download('averaged_perceptron_tagger_eng')  # Download English POS tagger

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
# POS mapping function 
# POS tags can be: ADJ (adjective), ADV (adverb), NOUN (noun), VERB (verb), etc
def get_wordnet_pos(tag):
    # Determine the WordNet POS tag based on the first letter of the input tag
    if tag.startswith('J'):
        return wordnet.ADJ  # Adjective
    elif tag.startswith('V'):
        return wordnet.VERB  # Verb
    elif tag.startswith('N'):
        return wordnet.NOUN  # Noun
    elif tag.startswith('R'):
        return wordnet.ADV  # Adverb
    else:
        return wordnet.NOUN  # Default to Noun if no match


def lemmatize_text(text):
    """
    Lemmatize text using WordNet lemmatizer with POS tagging.

    This version prints each change along with the POS tag of the changed word.
    """
    # Convert the input text to a string to ensure compatibility
    original_text = str(text)
    # Split the text into individual words
    words = original_text.split()
    # Obtain Part of Speech (POS) tags for each word
    pos_tags = pos_tag(words)

    # Initialize lists to store lemmatized words and any changes
    lemmatized_words = []
    changes = []

    # Iterate over each word and its POS tag
    for word, tag in pos_tags:
        # Map the POS tag to a WordNet POS tag
        wn_tag = get_wordnet_pos(tag)
        # Lemmatize the word using the mapped POS tag
        lemma = lemmatizer.lemmatize(word, wn_tag)

        # Check if the lemmatized word is different from the original
        if lemma != word:
            # Record the change if a difference is found
            changes.append((word, lemma, tag))
        # Add the lemmatized word to the list
        lemmatized_words.append(lemma)

    # Join the lemmatized words back into a single string
    result = ' '.join(lemmatized_words)

    # Print only if there were changes
    if changes:
        print(f"\nOriginal: {original_text}")
        print(f"Lemmatized: {result}")
        for original, lemma, pos in changes:
            print(f"  - {original} → {lemma}  (POS: {pos})")

    return result
# Apply lemmatization to all summaries
df_summary = df_summary.apply(lemmatize_text)
# Visualize
## Word Cloud for positive sentiments
from wordcloud import WordCloud

# Filter summaries for df_score >= 4
filtered_summaries = df_summary[df_score >= 4]

# Combine all filtered summaries into a single string
all_summaries = " ".join(str(summary) for summary in filtered_summaries)

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_summaries)

# Clear the memory
all_summaries = 0

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
## Word Cloud for negative sentiments
from wordcloud import WordCloud

# Filter summaries for df_score 1
filtered_summaries = df_summary[df_score == 1]

# Combine all filtered summaries into a single string
all_summaries = " ".join(str(summary) for summary in filtered_summaries)

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_summaries)

# Clear the memory
all_summaries = 0

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# ===============================
# >>> SECTION: 03 EMBEDDINGS
# ===============================
# If you get an error when loading Word2Vec, install scipy==1.12
# !pip install scipy==1.12
# Word Vectors
This notebook is focused on demonstrating various text representation techniques used in Natural Language Processing. It starts by defining a simple text corpus with sentences about different text representation methods like OneHot vectors, Bag of Words, TF-IDF, N-grams, and Word Embeddings. The corpus is then split into sentences and further into words, with all text converted to lowercase. A vocabulary of unique words is created, and each word is represented as a binary vector using one-hot encoding. The Bag of Words model counts the frequency of each word in the corpus using CountVectorizer from sklearn. TF-IDF is applied to the corpus using TfidfVectorizer from sklearn, which scales word frequencies by their importance across documents. N-grams are generated using CountVectorizer with an n-gram range of two, capturing pairs of consecutive words in the corpus. Finally it presents the Transformers library.
# Define a Corpus
my_corpus = f"""
    OneHot vectors are binary vectors.
    
    Bag of Words Counts words.
    
    TFIDF Counts words and weights words by importance.
    
    Ngrams Captures words sequences.
    
    Words Embeddings with Dense vectors.
    """
my_corpus
# Split the corpus into sentences by using '.' as a delimiter and remove the last empty element
my_corpus = my_corpus.split('.')[:-1]
# Strip leading and trailing whitespace from each sentence and filter out any empty sentences
my_corpus = [sentence.strip() for sentence in my_corpus if sentence]
my_corpus
# Tokenize the corpus
tokenized_corpus = [doc.lower().split() for doc in my_corpus]
tokenized_corpus
# 1-Hot Encoding
# Get all unique words
# Extract all unique words from the tokenized corpus and sort them
all_words = sorted(list(set(word for doc in tokenized_corpus for word in doc)))
# Print the size of the vocabulary
print(f"Vocabulary size: {len(all_words)}")
# Print the list of all unique words
print(all_words)
import numpy as np
import pandas as pd

# Create an identity matrix where each row represents a one-hot encoded vector for each unique word
one_hot_word_vectors = np.eye(len(all_words))

# Convert the one-hot encoded vectors into a DataFrame for better readability, using unique words as column headers
one_hot_word_vectors_df = pd.DataFrame(one_hot_word_vectors, columns=all_words)

# Display the DataFrame containing one-hot encoded vectors
one_hot_word_vectors_df
# Initialize a zero matrix to store one-hot encoded vectors for each *document* in the corpus
corpus_vectors = np.zeros((len(tokenized_corpus), len(all_words)))

# Iterate over each document and its index in the tokenized corpus
for i, doc in enumerate(tokenized_corpus):
    # Iterate over each word in the document
    for word in doc:
        # Iterate over each word and its index in the list of all unique words
        for j, w in enumerate(all_words):
            # If the word matches the unique word, set the corresponding position in the matrix to 1
            if w == word:
                corpus_vectors[i, j] = 1

# Convert the matrix of one-hot encoded vectors into a DataFrame for better readability
corpus_vectors_df = pd.DataFrame(corpus_vectors, columns=all_words)

# Display the DataFrame containing one-hot encoded vectors for the corpus
corpus_vectors_df
corpus_vectors_df.shape
# Bag of Words
# Create a Bag of Words representation 
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the CountVectorizer
count_vectorizer = CountVectorizer()

# Fit and transform the corpus
bow_matrix = count_vectorizer.fit_transform(my_corpus)

# Convert to DataFrame for better visualization
bow_df = pd.DataFrame(
    bow_matrix.toarray(),
    columns=count_vectorizer.get_feature_names_out()
)

# Display the Bag of Words representation
print("Bag of Words representation:")
bow_df
bow_df.shape
# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the corpus
tfidf_matrix = tfidf_vectorizer.fit_transform(my_corpus)

# Convert to DataFrame for better visualization
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)

# Display the TF-IDF representation
print("TF-IDF representation:")
tfidf_df
# N-grams
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the CountVectorizer with n-gram range
# ngram_range : is a tuple of two integers (min_n, max_n)
ngram_vectorizer = CountVectorizer(ngram_range=(2, 2))

# Fit and transform the corpus
ngram_matrix = ngram_vectorizer.fit_transform(my_corpus)
ngram_vectorizer.get_feature_names_out()
# Convert to DataFrame for better visualization
ngram_df = pd.DataFrame(
    ngram_matrix.toarray(),
    columns=ngram_vectorizer.get_feature_names_out()
)

# Display the N-grams representation
print("N-grams shape:", ngram_matrix.shape)
ngram_df
# Dense Word Embeddings (Word2Vec)
from gensim.models import Word2Vec

# Initialize the Word2Vec model
word2vec_model = Word2Vec(tokenized_corpus, # The corpus to train the model on
                          vector_size=100, # The dimensionality of the vectors
                          window=5, # The window size for the context window
                          epochs=5, # The number of epochs to train the model
                          min_count=1, # The minimum number of times a word must appear in the corpus to be included in the model
                          workers=4) # The number of threads to use for training

# Get the word embeddings
word_embeddings = word2vec_model.wv

# Print the word vectors
print(word_embeddings)
# Let's see the dimensionality of the vectors
print("\nVector dimensionality:", word_embeddings.vector_size)
# Let's see the vector for a specific word
my_word = "words"
print(f"\nVector for '{my_word}':")
print(word_embeddings[my_word])
# We can also find similar words
print(f"\nWords similar to '{my_word}':")
similar_words = word_embeddings.most_similar(my_word, topn=len(word_embeddings.index_to_key))
for idx, (word, similarity) in enumerate(similar_words):
    print(f"{idx+1}. {word}: {similarity:.4f}")
print(word_embeddings.index_to_key)
# Plotly heatmap
import plotly.express as px
def visualize_similarity_matrix(similarity_df):
    fig = px.imshow(similarity_df, labels=dict(x="Words", y="Words", color="Similarity"), x=similarity_df.columns, y=similarity_df.index, color_continuous_scale="Viridis")
    fig.update_layout(title="Word Similarity Matrix", xaxis_tickangle=-45, width=800, height=800)
    fig.show()
# Similarity matrix
# Create a similarity matrix manually since KeyedVectors doesn't have similarity_matrix attribute
import numpy as np
words = word_embeddings.index_to_key
similarity_matrix = np.zeros((len(words), len(words)))

for i, word1 in enumerate(words):
    for j, word2 in enumerate(words):
        if word1 != word2:
            similarity_matrix[i, j] = word_embeddings.similarity(word1, word2)

# Create a DataFrame for the similarity matrix
similarity_df = pd.DataFrame(similarity_matrix, index=words, columns=words)
visualize_similarity_matrix(similarity_df)
def visualize_2d_plot(df):
    # Create a scatter plot using Plotly
    fig = px.scatter(df, x='C1', y='C2', text='doc', title='Visualization of Word Embeddings', labels=["Component 1", "Component 2"])

    # Improve the layout
    fig.update_traces(textposition='top center', marker=dict(size=10, opacity=0.8), mode='markers+text')
    fig.update_layout(width=900, height=700, xaxis=dict(title='Component 1'), yaxis=dict(title='Component 2'))

    # Show the plot
    fig.show()
# PCA plot
from sklearn.decomposition import PCA

# Initialize PCA with 2 components
pca = PCA(n_components=2)

# Fit PCA on the word embeddings
pca.fit(word_embeddings.vectors)

# Transform the word embeddings using PCA
word_embeddings_2d = pca.transform(word_embeddings.vectors)

# Create a DataFrame for the 2D embeddings
pca_df = pd.DataFrame(
    word_embeddings_2d,
    columns=['C1', 'C2']
)
pca_df['doc'] = words
visualize_2d_plot(pca_df)
# TSNE plot
# t-SNE tries to preserve local relationships, not the global structure. For a small number of points (e.g., ~20 words), t-SNE often:
# Overemphasizes tiny distances
# Distorts distances between points not in a neighborhood
# Gives unpredictable layouts that "feel random"
from sklearn.manifold import TSNE

# Initialize TSNE with 2 components
tsne = TSNE(n_components=2, random_state=42)

# Fit and transform the word embeddings
# Set perplexity to a value less than the number of samples
# The perplexity is the number of samples in a neighborhood of a selected point
# Default perplexity is 30, so we need to reduce it if we have fewer than 30 samples
n_samples = word_embeddings.vectors.shape[0]
perplexity = min(30, n_samples - 1)  # Ensure perplexity is less than n_samples
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init='pca')
word_embeddings_2d = tsne.fit_transform(word_embeddings.vectors)

# Create a DataFrame for the 2D embeddings
tsne_df = pd.DataFrame(
    word_embeddings_2d,
    columns=['C1', 'C2']
)
tsne_df['doc'] = words

visualize_2d_plot(tsne_df)
# Compute the embeddings of the sentences in the corpus from the word embeddings
# Initialize an empty list to store sentence embeddings
sentence_embeddings = np.zeros((len(tokenized_corpus), word_embeddings.vector_size))
# Iterate through each document in the corpus
for i, doc in enumerate(tokenized_corpus):
    # Initialize a numpy array of zeros for the sentence vector
    sentence_vector = np.zeros(word_embeddings.vector_size)
    word_count = 0
    
    # Iterate through each word and add its vector to the sentence vector
    for word in doc:
        if word in word_embeddings:
            sentence_vector += word_embeddings[word]
            word_count += 1
    
    # If we found words in the model, calculate the average
    if word_count > 0:
        sentence_vector = sentence_vector / word_count
    
    # Add the sentence embedding to our list
    sentence_embeddings[i] = sentence_vector
# Create a DataFrame with the sentence embeddings
# The error occurs because word_embeddings.index_to_key has 17 items but our vectors have 100 dimensions
# We need to create column names that match the dimensions of our vectors
sentence_embeddings_df = pd.DataFrame(
    sentence_embeddings,
    columns=[f"dim_{i}" for i in range(word_embeddings.vector_size)]
)
sentence_embeddings_df
# Similarity matrix for document embeddings
# Create a similarity matrix manually since KeyedVectors doesn't have similarity_matrix attribute
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = np.zeros((len(my_corpus), len(my_corpus)))
for i, embedding_i in enumerate(sentence_embeddings):
    for j, embedding_j in enumerate(sentence_embeddings):
        if i != j:
            similarity_matrix[i, j] = cosine_similarity(embedding_i.reshape(1, -1), embedding_j.reshape(1, -1))[0, 0]

# Create a DataFrame for the similarity matrix
doc_names = ["doc_" + str(i+1) for i in range(len(tokenized_corpus))]
similarity_df = pd.DataFrame(similarity_matrix, index=doc_names, columns=doc_names)
visualize_similarity_matrix(similarity_df)

# Visualize PCA for document embeddings
pca = PCA(n_components=2)

# Fit PCA on the word embeddings
pca.fit(sentence_embeddings)

# Transform the word embeddings using PCA
sentence_embeddings_2d = pca.transform(sentence_embeddings)

# Create a DataFrame for the 2D embeddings
pca_df = pd.DataFrame(
    sentence_embeddings_2d,
    columns=['C1', 'C2']
)
pca_df['doc'] = ["doc_" + str(i) for i in range(len(tokenized_corpus))]
visualize_2d_plot(pca_df)
# Transformers
## Environmental Variables
we will need to use Environment Variables:
- HF_TOKEN is you huggingface token, you may generate one on this url: https://huggingface.co/settings/tokens

## On Linux do:
- `nano ~/.bashrc`
- `export HF_TOKEN="..."`
- `source ~/.bashrc`
- `echo $HF_TOKEN`
import os
os.environ['HF_TOKEN'] = "your huggingface token here"
os.environ["HF_HOME"] = r"C:\my_hf_models"
# https://huggingface.co/sentence-transformers
# !pip install -U sentence-transformers
# https://pytorch.org/get-started/locally/
# !pip3 install torch
from sentence_transformers import SentenceTransformer
import torch
if torch.cuda.device_count()>0:
    my_device = "cuda"
    print(f"You have {torch.cuda.device_count()} GPUs available.")
else:
    my_device = "cpu"
    print("You have no GPUs available. Running on CPU.")
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',
                                       token=os.environ["HF_TOKEN"],
                                       cache_folder=os.environ["HF_HOME"],
                                       device=my_device)
# Encode the corpus using the embeddings model
word_embeddings_transformer = embeddings_model.encode(my_corpus)

# Print the shape of the resulting embeddings
print(word_embeddings_transformer.shape)

# Output the embeddings
word_embeddings_transformer
# Initialize a zero matrix to store similarity scores between documents
similarity_matrix = np.zeros((len(my_corpus), len(my_corpus)))

# Iterate over each pair of embeddings to compute cosine similarity
for i, embedding_i in enumerate(word_embeddings_transformer):
    for j, embedding_j in enumerate(word_embeddings_transformer):
        # Avoid computing similarity of a document with itself
        if i != j:
            # Compute and store the cosine similarity between different document embeddings
            similarity_matrix[i, j] = cosine_similarity(embedding_i.reshape(1, -1), embedding_j.reshape(1, -1))[0, 0]

# Create a DataFrame for the similarity matrix
doc_names = ["doc_" + str(i+1) for i in range(len(tokenized_corpus))]
similarity_df = pd.DataFrame(similarity_matrix, index=doc_names, columns=doc_names)
visualize_similarity_matrix(similarity_df)
## Test Embeddings - related words
# Define a list of words to analyze
word_list = ["book", "book!", "publication", "article"]

# Encode the list of words using the embeddings model
word_embeddings_transformer = embeddings_model.encode(word_list)

# Calculate the cosine similarity matrix for the encoded words
cosine_similarities = cosine_similarity(word_embeddings_transformer)

# Print the cosine similarity matrix
print("Cosine Similarity Matrix:")
print(cosine_similarities)

# Create a DataFrame from the cosine similarity matrix for better visualization
similarity_df = pd.DataFrame(cosine_similarities, index=word_list, columns=word_list)

# Visualize the similarity matrix
visualize_similarity_matrix(similarity_df)
## Calculate normalized mean values of embeddings
# Calculate the mean of the absolute values of the embeddings along axis 1
mean_embeddings = np.mean(np.abs(word_embeddings_transformer), axis=1)
print("Normalized Mean values of embeddings:", mean_embeddings)

# Calculate the standard deviation of the embeddings along axis 1
std_embeddings = np.std(word_embeddings_transformer, axis=1)
print("Standard Deviation of embeddings:", std_embeddings)

# Calculate the norm of the embeddings along axis 1
norm_embeddings = np.linalg.norm(word_embeddings_transformer, axis=1)
print("Norm of embeddings:", norm_embeddings)
## Generate random vectors with the same mean and std
# Generate random vectors with the same mean and standard deviation as the word embeddings
random_vectors = np.random.normal(loc=np.mean(word_embeddings_transformer),
                                  scale=np.std(word_embeddings_transformer),
                                  size=word_embeddings_transformer.shape)

# Calculate and print the normalized mean values of the random vectors
mean_random_vectors = np.mean(np.abs(random_vectors), axis=1)
print("Normalized Mean values of random vectors:", mean_random_vectors)

# Calculate and print the standard deviation of the random vectors
std_random_vectors = np.std(random_vectors, axis=1)
print("Standard Deviation of random vectors:", std_random_vectors)

# Calculate and print the norm of the random vectors
norm_random_vectors = np.linalg.norm(random_vectors, axis=1)
print("Norm of random vectors:", norm_random_vectors)
# Print the cosine similarity matrix for the random vectors
print("Cosine Similarity Matrix random vectors:")
cosine_similarities = cosine_similarity(random_vectors)

# Display the cosine similarity matrix
print(cosine_similarities)

# Create a DataFrame for the cosine similarity matrix with appropriate labels
similarity_df = pd.DataFrame(
    cosine_similarities, 
    index=["Random Vector 1", "Random Vector 2", "Random Vector 3", "Random Vector 4"], 
    columns=["Random Vector 1", "Random Vector 2", "Random Vector 3", "Random Vector 4"]
)

# Visualize the similarity matrix
visualize_similarity_matrix(similarity_df)

## car ~ vehicle + motorcycle - bike
# Define a list of words to analyze
sentences = ["car", "vehicle", "motorcycle", "bike"]

# Encode the words into embeddings using the embeddings model
embeddings = embeddings_model.encode(sentences)

# Calculate the cosine similarity between the embedding of "car" and the vector operation (vehicle + motorcycle - bike)
similarity_score = cosine_similarity(
    embeddings[0].reshape(1, -1), 
    (embeddings[1] + embeddings[2] - embeddings[3]).reshape(1, -1)
)[0, 0]

# Print the similarity score
print(similarity_score)
## Greece ~ Athens + Italy - Rome
# Define the list of words for the analogy task
sentences = ["Greece", "Athens", "Italy", "Rome"]

# Encode the words into embeddings
embeddings = embeddings_model.encode(sentences)

# Calculate and print the cosine similarity for the analogy: Greece ~ Athens + Italy - Rome
similarity_score = cosine_similarity(
    embeddings[0].reshape(1, -1), 
    (embeddings[1] + embeddings[2] - embeddings[3]).reshape(1, -1)
)[0, 0]
print(similarity_score)
So embeddings work!
## Sentence embeddings
my_sentences = [
    # Interrelated sentences - group 1
    "The data is preprocessed to remove noise and outliers.",
    "Noise and outliers are eliminated during data preprocessing.",
    "Preprocessing cleans the data by filtering out noise and irregularities.",

    # Interrelated sentences - group 2
    "Paris is the capital of France.",
    "Athens is the capital of Greece.",
    "Rome is the capital of Italy."
]
my_embeddings = embeddings_model.encode(my_sentences)
similarity_matrix = cosine_similarity(my_embeddings)
print(similarity_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=my_sentences, columns=my_sentences)
visualize_similarity_matrix(similarity_df)
## Tokenizers
from transformers import AutoTokenizer
my_model = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(my_model,
                                          token=os.environ["HF_TOKEN"],
                                          cache_dir=os.environ["HF_HOME"])
https://huggingface.co/learn/llm-course/en/chapter6/5

In LLaMA and similar Byte-Pair Encoding (BPE) based models:

> **tokens ≠ words** (exactly),

> **tokens ≈ pieces of words + punctuation + space markers**

This helps the model handle any language efficiently with a smaller vocabulary.

**Example of subwords**

Take the word:
`unbelievable`

A tokenizer might split it like this:

```
['un', 'believ', 'able']
```

* `"un"` → a common prefix
* `"believ"` → root of "believe", "believer", etc.
* `"able"` → a common suffix

This way, even if `"unbelievable"` was never seen during training, the model knows the meaning from its parts.

---
In the follwoing example:

words: "Hello", "world", "Let", "tokenize", "this", "text"

punctuation: ",", "!", ".", "'s"

space indicators: the Ġ marks the start of a new word with a space. The Ġ symbol is not a space itself, but it indicates that a space precedes the token. This is a convention used in the LLaMA tokenizer (and some others like RoBERTa).

Hence `','` and `'Ġ,'` **are different tokens** in LLaMA-style or BPE-style tokenizers.
tokens = tokenizer.tokenize("Hello, world! Let's tokenize this text.")
print(tokens)


# ===============================
# >>> SECTION: 04 SUPERVISED LEARNING
# ===============================
# Import Dataset
df_summary = 0
df_score = 0
# This code is downloading the notebook from GitHub and running it
import requests
from pathlib import Path
url = "https://raw.githubusercontent.com/nbakas/NLP/refs/heads/main/02-Preprocessing.ipynb"
filename = url.split("/")[-1]
local_path = Path.cwd() / filename
if not local_path.exists():
    response = requests.get(url)
    response.raise_for_status()
    local_path.write_bytes(response.content)
%run $local_path
df_summary
df_score
# Embeddings
## TF-IDF
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_summary, df_score, test_size=0.2, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_features=384,  # Limit features to reduce dimensionality
    min_df=5,           # Minimum document frequency
    max_df=0.8,         # Maximum document frequency (ignore terms that appear in >80% of documents)
    stop_words='english'
)

# Fit and transform the training data
X_train = tfidf_vectorizer.fit_transform(X_train)
X_train[:5].toarray()
import numpy as np
np.mean(X_train.toarray())
import random

# Get the number of rows in X_train
num_rows = X_train.shape[0]

# Generate 10 random indices
random_indices = random.sample(range(num_rows), 10)

# Print 10 random lines of X_train
for idx in random_indices:
    print(f"Index: {idx}, Mean: {np.mean(X_train[idx].toarray()[0], axis=0)}")
# Transform the test data
X_test = tfidf_vectorizer.transform(X_test)
# Display the shape of the TF-IDF matrices
print(f"Training TF-IDF matrix shape: {X_train.shape}")
print(f"Testing TF-IDF matrix shape: {X_test.shape}")
# Get feature names (words)
feature_names = tfidf_vectorizer.get_feature_names_out()
print(f"Number of features (words): {len(feature_names)}")
print(f"Features: {feature_names}")
# Most frequent words
import numpy as np
import pandas as pd

# Get the sum of TF-IDF values for each term across all documents
tfidf_means = np.array(X_train.mean(axis=0)).flatten()

# Create a DataFrame with terms and their TF-IDF sums
term_importance = pd.DataFrame({
    'term': feature_names,
    'tfidf_mean': tfidf_means
})

# Sort by importance (TF-IDF sum)
term_importance = term_importance.sort_values('tfidf_mean', ascending=False)

# Display the top 10 most important terms
print("Top 10 most important terms:")
print(term_importance.head(10))
## Word2Vec
X_train, X_test, y_train, y_test = train_test_split(df_summary, df_score, test_size=0.2, random_state=42)
# Tokenize the text data for Word2Vec
tokenized_train = [text.split() for text in X_train]
tokenized_test = [text.split() for text in X_test]
tokenized_train[:10]
from gensim.models import Word2Vec
# Define the Word2Vec model
w2v_model = Word2Vec(
    sentences=tokenized_train,
    vector_size=384, # Dimensionality of the word vectors
    window=5, # Maximum distance between the current and predicted word within a sentence
    min_count=2, # Ignores words with frequency lower than this
    workers=4, # Number of threads to run in parallel
    sg=1, # Training algorithm: 1 for skip-gram; 0 for CBOW
    seed=42
)
print("Training Word2Vec model...")
# Train the model
w2v_model.train(
    tokenized_train, # List of sentences to train
    total_examples=len(tokenized_train), # Number of sentences to train on
    epochs=10 # Number of epochs 
)
print(f"Vocabulary size: {len(w2v_model.wv.key_to_index)}")
# Function to create document vectors by averaging word vectors
def document_vector(doc, model):
    # Filter words that are in the model vocabulary
    doc_words = [word for word in doc if word in model.wv]
    if len(doc_words) == 0:
        # Return zeros if no words are in vocabulary
        return np.zeros(model.vector_size)
    # Return the mean of all word vectors in the document
    return np.mean([model.wv[word] for word in doc_words], axis=0)
# Create document vectors for training and testing sets
X_train = np.array([document_vector(doc, w2v_model) for doc in tokenized_train])
X_test = np.array([document_vector(doc, w2v_model) for doc in tokenized_test])
X_train
print(f"Training Word2Vec matrix shape: {X_train.shape}")
print(f"Testing Word2Vec matrix shape: {X_test.shape}")
# Explore some word similarities
my_test_word = "delicious" # Try another common word e.g. food, price, service, etc.
try:
    # Find words most similar
    print(f"\nWords most similar to '{my_test_word}':")
    for word, similarity in w2v_model.wv.most_similar(my_test_word, topn=5):
        print(f"{word}: {similarity:.4f}")
except KeyError:
    print(f"Word '{my_test_word}' not in vocabulary. Try another common word.")
## Transformers Embeddings
from sentence_transformers import SentenceTransformer
import os
import torch
import os
if torch.cuda.device_count()>0:
    my_device = "cuda"
    print(f"You have {torch.cuda.device_count()} GPUs available.")
else:
    my_device = "cpu"
    print("You have no GPUs available. Running on CPU.")
embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',
                                       token=os.environ["HF_TOKEN"],
                                       cache_folder=os.environ["HF_HOME"],
                                       device=my_device)
#################################################################################################
#################################################################################################
########## The following cell will take some time (e.g. 20 min on the CPU of a laptop) ##########
#################################################################################################
#################################################################################################
word_embeddings_transformer = embeddings_model.encode(df_summary)
print(word_embeddings_transformer.shape)
word_embeddings_transformer
X_train, X_test, y_train, y_test = train_test_split(word_embeddings_transformer, df_score, test_size=0.2, random_state=42)
print(f"Training Word2Vec matrix shape: {X_train.shape}")
print(f"Testing Word2Vec matrix shape: {X_test.shape}")
# ML Models
## Logistic Regression
# Import necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score

# Initialize the Logistic Regression model
# Use 'multinomial' solver for multi-class classification
lr_model = LogisticRegression(
    multi_class='multinomial',  # Multinomial for multi-class problems
    solver='lbfgs',             # Efficient solver for multinomial logistic regression
    max_iter=1000,              # Increase max iterations to ensure convergence
    random_state=42,            # For reproducibility
    n_jobs=-1                   # Use all available cores
)
# Train the model
print("Training Logistic Regression model...")
lr_model.fit(X_train, y_train)
# Make predictions
y_pred = lr_model.predict(X_test)

# Evaluate the model
print("\nLogistic Regression Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# Display detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Confusion Matrix
plt.figure(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(set(y_test)), 
            yticklabels=sorted(set(y_test)))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Logistic Regression')
plt.show()
## Random Forest Classifier
# ~45 min!
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
# Make predictions
y_pred = rf_model.predict(X_test)
# Evaluate the model
print("\nLogistic Regression Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# Display detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
## Gradient Boosting (XGBoost)
from xgboost import XGBClassifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
# XGBoost expects classes to start from 0, but our labels are 1-5
# Convert labels from 1-5 to 0-4 for training, by subtracting 1
xgb_model.fit(X_train, y_train - 1)
# Make predictions
y_pred = xgb_model.predict(X_test)
# Evaluate the model
print("\nLogistic Regression Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test-1, y_pred):.4f}")
# Display detailed classification report
print("\nClassification Report:")
print(classification_report(y_test-1, y_pred))


# ===============================
# >>> SECTION: 06 LDA
# ===============================
# Load dataset and pre-processing
df_summary = 0
# This code is downloading the notebook from GitHub and running it
import requests
from pathlib import Path
url = "https://raw.githubusercontent.com/nbakas/NLP/refs/heads/main/02-Preprocessing.ipynb"
filename = url.split("/")[-1]
local_path = Path.cwd() / filename
if not local_path.exists():
    response = requests.get(url)
    response.raise_for_status()
    local_path.write_bytes(response.content)
%run {str(local_path)}
df_summary
# Libraries
# We will use gensim library for topic modeling
# Import corpora module for document processing
from gensim import corpora
# Import LdaMulticore for parallel LDA implementation
from gensim.models.ldamulticore import LdaMulticore
# Import matplotlib for visualization
import matplotlib.pyplot as plt
# Convert df_summary to list of texts
my_texts = df_summary.astype(str).tolist()
my_texts[:5]
# Tokenize the texts
processed_texts = [my_text.split() for my_text in my_texts]
processed_texts[:5]
# Create a dictionary mapping words to their IDs
# This cell is creating a dictionary of words and their IDs from the processed texts.
my_dictionary = corpora.Dictionary(processed_texts)
my_dictionary
# Print 10 random items from the dictionary to understand its structure
print("10 random items from the dictionary:")
import random
random_ids = random.sample(list(my_dictionary.keys()), 10)
for word_id in random_ids:
    print(f"Word ID {word_id}: {my_dictionary[word_id]}")
len(my_dictionary)
# Filter out extreme values
# Filter out extreme values (optional) to improve LDA performance and quality
# no_below=100: Remove words that appear in fewer than 100 documents (rare terms)
#   - Removes noise and very specific terms that don't help identify general topics
# no_above=0.1: Remove words that appear in more than 10% of documents (too common)
#   - Removes overly common words that appear across many topics and don't help differentiate
# This filtering reduces my_dictionary size, speeds up computation, and helps LDA focus on meaningful topic-specific words
my_dictionary.filter_extremes(no_below=10, no_above=0.1)
my_dictionary
len(my_dictionary)
# Create a document-term matrix
# This cell is creating a "bag-of-words" representation of your processed texts using the Gensim library.
# In the following line, `my_corpus = [my_dictionary.doc2bow(text) for text in processed_texts]`, each document in `processed_texts` is converted to a bag-of-words format using `doc2bow()`.
my_corpus = [my_dictionary.doc2bow(text) for text in processed_texts]
my_corpus[:10]
# Each tuple (word_id, frequency) represents a word by its dictionary ID and how many times it appears in that document.
processed_texts[:10]
# print the first 10 documents in the corpus, their words and their IDs
for doc_id, doc in enumerate(my_corpus[:10]):
    print(f"Document {doc_id+1}:")
    for word_id, freq in doc:
        word = my_dictionary[word_id]
        print(f"  Word ID {word_id} ('{word}'): Frequency {freq}")
---

**Note on Test Set Evaluation**

Since **LDA is an unsupervised learning method**, we often use the full dataset to build the model — there are no labels to predict, as in supervised learning.

However, using a **test set** can still be very useful.
It helps:

* Evaluate how well the model **generalizes to new, unseen data**
* Compare models with different **numbers of topics**
* Avoid **overfitting** to the training data

---
# Set LDA parameters
num_topics = 10  # Number of topics to be extracted
my_passes = 10 # Number of my_passes of the corpus through the model during training. More my_passes means better accuracy but longer runtime
workers = 4  # Number of worker processes for parallel computing
# Train the LDA model
# It will take ~10 minutes to train the model if the dictionary is not filtered.
# https://radimrehurek.com/gensim/models/ldamulticore.html
lda_model = LdaMulticore(
    corpus=my_corpus, # The document-term list we created earlier
    id2word=my_dictionary, # Maps word IDs to actual words for interpretable output
    num_topics=num_topics, # Number of topics to extract 
    passes=my_passes, # Number of training my_passes through the corpus 
    workers=workers, # Number of parallel processes to use 
    alpha='symmetric', # Topic distribution prior - 'symmetric' gives equal probability to all topics initially
    eta='auto' # Word distribution prior (influences how words are distributed across topics). 'auto' lets the model learn optimal word weights. β in notes.
)
# Evaluate LDA model performance
## Coherence score
- A coherence score tells us if the words in a topic appear in similar texts or contexts.
- For example, in a good topic like: ["dog", "cat", "pet", "animal"], these words often show up in the same documents.
- It takes values between 0 and 1, with 1 being the highest coherence. Typical values are between 0.3 and 0.6.
# https://radimrehurek.com/gensim/models/coherencemodel.html
from gensim.models.coherencemodel import CoherenceModel
coherence_model_lda = CoherenceModel(model=lda_model, # LDA model
                                     texts=processed_texts, # list of texts, each text is a list of words
                                     dictionary=my_dictionary, # dictionary of words and their IDs
                                     coherence='c_v', # coherence measure, c_v id defined as the average pointwise mutual information of all word pairs in a topic
                                     topn=20 # number of words to consider for coherence score
                                     )
coherence_score = coherence_model_lda.get_coherence()
print(f"Coherence Score: {coherence_score:.4f}")
## Perplexity 
**Perplexity** is a measure of how well a topic model can **explain the words** in a new document.

* It uses the **words in the document** to guess which topics are present.
* Then it checks how **likely those words** are, based on the **learned topic-word probabilities**.
* It does **not care about word order** — only which words appear and how often.

**Lower perplexity = better fit**
(The model is less “surprised” by the document’s words.)

Even if the document has **multiple topics**, the perplexity can still be low — as long as the topics match well with the document’s words.

---
- Perplexity is the exponential of the negative average log-likelihood per word
- Typical perplexity values for LDA models are usually in the range of 100–1000
- Lower values (e.g., < 100) indicate better generalization (less surprise),
- However, very low perplexity on the training set (e.g., < 50) can be a sign of overfitting,
meaning the model fits the training data too closely and may not generalize well to unseen data
- Very high values (e.g., > 1000) suggest poor topic modeling or an inappropriate number of topics
  
**Perplexity Formula:**

If `log_perplexity` is the negative average log-likelihood per word (from Gensim):

Perplexity = e^(-log_perplexity)

Where:
- `log_perplexity` is returned by `lda_model.log_perplexity(corpus)`
- `exp` is the exponential function (base *e*)
| log_perplexity | Actual perplexity | Interpretation                  |
|----------------|-------------------|---------------------------------|
| -5             | ~148              | Very good fit                   |
| -6             | ~403              | Good                            |
| -7             | ~1097             | Acceptable to borderline high   |
| -8             | ~2980             | Likely too high → poor generalization |
perplexity = lda_model.log_perplexity(my_corpus)
print(f"Perplexity: {perplexity:.4f}")
# Visualize LDA topics using pyLDAvis
# !pip install pyLDAvis
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# Prepare the visualization
vis_data = gensimvis.prepare(lda_model, my_corpus, my_dictionary)

# Set the figure size for better visualization
pyLDAvis.enable_notebook()

# Display the interactive visualization
pyLDAvis.display(vis_data)
# Get 10 random documents and print their topics
import random
import numpy as np

# Select 10 random document indices
random_doc_indices = random.sample(range(len(my_corpus)), 10)

print("\nTopic Distribution for 10 Random Documents:")
print("-" * 50)

for idx in random_doc_indices:
    # Get the document's topic distribution
    doc_topics = lda_model.get_document_topics(my_corpus[idx])
    
    # Sort topics by probability (highest first)
    doc_topics = sorted(doc_topics, key=lambda x: x[1], reverse=True)
    
    # Get the original text (if available)
    original_text = df_summary.iloc[idx]
    
    print(f"\nDocument {idx}: \"{original_text}\"")
    print("Topic Distribution:")
    
    for topic_id, prob in doc_topics[:3]:
        # Get the top words for this topic
        topic_words = lda_model.show_topic(topic_id, topn=5)
        words = ", ".join([word for word, _ in topic_words])
        
        # Format the probability as a percentage
        prob_percent = prob * 100
        
        print(f"  Topic {topic_id+1}: {prob_percent:.2f}% ({words})")


# ===============================
# >>> SECTION: MIDTERM EXAM (Airline Sentiment)
# ===============================
## Midterm Exam - NLP - Spring 2024

### General Instructions
- Answer ALL mandatory questions.
- Time Allocated: 3 hours
- This is an individual examination; you are NOT allowed to receive help from your peers.
- You are free to use class notes, notebooks and the internet to refer to online documentation and examples, including usage of existing code.
- Use of ChatGPT is strictly prohibited.


#### Good Luck!!!

<br>

### **Problem Statement**

<br>

## Intro

<br>

<img src = "https://www.surveysensum.com/wp-content/uploads/2020/02/SENTIMENT-09-1.png">

<br>

A sentiment analysis job about the problems of each major U.S. airline. Twitter data was scraped from February of 2015 and contributors were asked to first classify positive, negative, and neutral tweets, followed by categorizing negative reasons (such as "late flight" or "rude service").

For example, it contains whether the sentiment of the tweets in this set was positive, neutral, or negative for six US airlines:

The information of main attributes for this project as follows;

* **`airline_sentiment`** : Sentiment classification.(positivie, neutral, and negative)
* **`negativereason`** : Reason selected for the negative opinion
* **`airline`** : Name of 6 US Airlines('Delta', 'United', 'Southwest', 'US Airways', 'Virgin America', 'American')
* **`text`** : Customer's opinion

<br>

**Objective**

You are given a dataset of US Airline tweets and their sentiment. The task is to do sentiment analysis about the problems of each major U.S. airline. Twitter data was scraped from February of 2015 and contributors were asked to first classify positive, negative, and neutral tweets, followed by categorizing negative reasons (such as "late flight" or "rude service").


### **Load the libraries**
# Install the pyLDAvis library
!pip install pyLDAvis
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
### Load the dataset

Twitter data was scraped from February of 2015 and contributors were asked to first classify positive, negative, and neutral tweets, followed by categorizing negative reasons (such as "late flight" or "rude service"). The information of some of the main attributes for this project are as follows:

* **`airline_sentiment`** : Sentiment classification.(positivie, neutral, and negative)
* **`negativereason`** : Reason selected for the negative opinion
* **`airline`** : Name of 6 US Airlines('Delta', 'United', 'Southwest', 'US Airways', 'Virgin America', 'American')
* **`text`** : Customer's opinion
# 1. Load the dataset from https://raw.githubusercontent.com/satyajeetkrjha/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv
# using as normal the pandas .read_csv() function. Store in a new variable named 'data'
# 2. Print the dimensionality of data
# 3. Preview the first few rows of data

data = pd.read_csv('https://raw.githubusercontent.com/satyajeetkrjha/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv')
print(data.shape)
data
# Get the overall information of the data DataFrame (data types, missing values, etc.)

data.info()
# Get all the unique airline names within 'data' and store them in a variable 'airlines'. Which feature would you use?
# This list qill be used later on in the analysis
# Print the 'airlines' results

print('Unique Airline Names:')
airlines = data['airline'].unique().tolist()
airlines
# 1. Set the column 'tweet_id' as the index of data (if you don't know how, you can drop it. Alternatively, the next step will sort this out).
# Remember to conduct this step inplace or with assignment back to "data" for any changes to take effect
# 2. Select and keep ONLY a subset of the features/columns: keep only 'text' and 'airline_sentiment' columns, and re-assign back to data (overwrite data)
# 3. Preview once more the results of data as a sanity check. Have your changes gone through?

data.set_index('tweet_id', inplace = True)
data
data = data[['text', 'airline_sentiment']]
data
# Drop the duplicates within data. Remember to assign back or conduct this step with replacement.
# Sanity check: print the dimensionality of 'data' before and after the drop. Was your drop successful?

data.shape
data = data.drop_duplicates()
data.shape
# Check for null values in data; you can return True/False, counts or any other solution of your choice to investigate for missing values

data.isnull().sum().sum() 
# Can you print the 'text' (column) of one random sample in data?

data['text'].sample(1)
# **EDA**
### **Counts per type of Sentiment**
# Get the number (frequency/count) of instances (samples) per each sentiment class (this will be a useful insight for the ML classifier later on).
# Store in a variable with a name of your choice and print the results

countpersentiment = data['airline_sentiment'].value_counts()
countpersentiment
# Use the variable above to plot a seaborn countplot

sns.countplot(x="airline_sentiment", data=data);
# - What do you observe?? Briefly describe here your findings
# we can easily observe that the majority of the comments that the airline receives are by far negative ones (around 9k), 
# followed by neutral ones (around 3k) and then followed by positive ones (around 2k)
### WordCloud : Keyword analysis
WordCloud is one of the easiest way to show which word mainly(frequently) appears in the set of sentences.

But it can be just one of pieces of visualization if there's no appropriate text preprocessing before drawing it.
#  Bonus activity: can you filter and draw a Wordcloud of only the positive sentiment cases? Use also STOPWORDS from WordCloud

df_pos = data[data['airline_sentiment'] == "positive"]

wordcloud = WordCloud(stopwords = STOPWORDS,max_words = 1000 , width = 1600 , height = 800,
                     collocations=True).generate(" ".join(df_pos['final_text']))
plt.imshow(wordcloud)
Wordcloud for Negative sentiments of tweets
#  Bonus activity: can you filter and draw a Wordcloud of only the negative sentiment cases? Use also STOPWORDS from WordCloud

df_neg = data[data['airline_sentiment'] == "negative"]

wordcloud = WordCloud(stopwords = STOPWORDS,max_words = 1000 , width = 1600 , height = 800,
                     collocations=True).generate(" ".join(df_neg['final_text']))
plt.imshow(wordcloud)

## **Text pre-processing of the tweet text data**
Now, we will clean the tweet text data and apply classification algorithms on it
# Define a function named 'lowercase' to lowercase your text in data

def lowercase(text):
    return text.lower() 
# Apply your 'lowercase' function to lowercase your data on the 'text' feature of data.
# Remember to reassign back to the same column for the changes to take effect.
# Preview your results

data['text'] = data['text'].apply(lowercase)
data['text'].head()
# Define a function named 'remove_stopwords' to remove stopwords from your text.
# BONUS / Extra points: you may also wish to append to your stopwords the names of the airlines that you had previously stored in the variable 'airlines'
# You may also consider pre-processing these values before appending them.
# Optionally, you can also append any other frequently encountered words in your stopwords.

def remove_stopwords(text):
    textArr = text.split(' ')
    rem_text = " ".join([i for i in textArr if i not in stop_words])
    return rem_text
# Apply your 'remove_stopwords' function to remove the stopwords from your 'text' feature of data.
# Remember to reassign back to the same column for the changes to take effect.
# Preview your results

data['text'] = data['text'].apply(remove_stopwords)
data['text'].head()
# Define one or more functions to perform ****at least one more additional**** pre-processing step of your choice based on the data that you observe!
# The cleaner the data, the higher the mark for this activity :D

def remove_usernames(text):
    if isinstance(text, str):
        text = re.sub(r'@[^\s]+', '', text)
    return text
# Justify and briefly describe your choice of pre-processing step(s) here

data['text'] = data['text'].apply(remove_usernames)
data['text'].head()

#i chose to remove the usernames in the comments because i didn't want to keep the mentioned names
# Create a list of the 'text' column from data. Store in a new variable named 'text_list'.
# Preview the first entry of your text_list

text_list = data['text'].values.tolist()
text_list[0]
type(text_list)
# Define a function to perform lemmatization, named 'lemmatization'.
# Bonus: allow postags only NOUN and ADJ

nlp = spacy.load('en_core_web_sm')
def lemmatization(text, allowed_postags=['NOUN', 'ADJ']): 
       output = []
       for sent in text:
            doc = nlp(sent) 
            output.append([token.lemma_ for token in doc if token.pos_ in allowed_postags ])
       return output
# Run your lemmatization function on text_list (remember, this is a list, not a DataFrame so you cannot use apply())
# Store in a new variable named 'tokenized_reviews'
### THIS STEP MAY TAKE A WHILE TO EXECUTE ###

tokenized_reviews = lemmatization(text_list)
tokenized_reviews
# Concatenate the tokens into single sentences once again, creating a new feature within data named 'final_text'

data['final_text']= ''

for i in range(len(text_list)):
    data['final_text'].iloc[i] = "".join(text_list[i])
data.head()
# **Vectorization**
# Store the final_text column from data as your variable X
# Store the airline_sentiment column from data as your class vector y

X = data['final_text']
y = data['airline_sentiment']
X
y
# In this step we want to map the classes 'positive', 'neutral' and 'negative' to numerical values
# Instantiate a LabelEncoder() object and store in a variable named 'le'
# Fit and transform your LabelEncoder to your y class variable

le = LabelEncoder()
y = le.fit_transform(y)
y
### The data is split in the standard ratio
# Use the holdout method from sklearn to split your data into train and test. Set the random_state to 42 and use stratification.
# Print the dimensionality of the end results as a sanity check

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, stratify = y)
print('X_train shape: ', X_train.shape)
print('X_test shape: ', X_test.shape)
# Import the library of the vectorization technique of your choice

from sklearn.feature_extraction.text import CountVectorizer
# Briefly justify your choice for this particular vectorizer

# i chose the “bags-of-words” representation which ignores structure and simply counts how often each word occurs. 
# CountVectorizer allows us to use the bags-of-words approach, by converting a collection of text documents into
# a matrix of token counts.
# Instantiate a vectorizer of your choice into a new variable named 'vect'

vect = CountVectorizer().fit(X_train)
vect
# Can you also create/ instantiate your vectorizer using unigrams and bigrams?
# Optional arguments to consider can be: min_df, max_df and max_features
# Store in a variable 'vect_grams'

vect_grams = CountVectorizer(min_df = 5, max_df = 20, ngram_range = (1,2)).fit(X_train)
# Apply any of the two afore-mentioned vectorizers to your train and test data to convert the text into numbers,
# and store the results in X_train_vec and X_test_vec respectively

X_train_vec = vect.transform(X_train)
X_test_vec = vect.transform(X_test)
# **Model Building**
# Import the Supervised Learning algorithm (classification model) of your choice

from sklearn.svm import LinearSVC
# 1. Instantiate your ML model using any hyperparameters of your choice (no need for tuning)
# 2. Fit your model to your train data
# 3. Predict your test data. Store into a variable named 'pred'
# 4. Report the accuracy score between your predicted and real test values

SVCmodel = LinearSVC()
SVCmodel.fit(X_train_vec, y_train)
y_pred = SVCmodel.predict(X_test_vec)
# Print the classification report

print(classification_report(y_test, y_pred))
# Build the confusion matrix between your predicted and real test values
# Store into a new variable named 'cm'

cm = confusion_matrix(y_test, y_pred)
cm
# Extra: plot in a heatmap the confusion matrix

f, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(cm, 
            annot=True, 
            annot_kws={'size': 8}, 
            cmap="Spectral_r");
# What do you observe in your results? Describe your findings

# we can see that the true mapped values to 0 and the the predicted mapped values to 0 have the highest correlation
# **Unsupervised Learning - Topic Modelling**
# Create a corpora.Dictionary by passing the tokenized_reviews. Store in a new variable 'dictionary'

dictionary = corpora.Dictionary(tokenized_reviews)

texts = tokenized_reviews
print(tokenized_reviews[1])
# Create the doc_term_matrix from your tokenized_reviews

doc_term_matrix = [dictionary.doc2bow(rev) for rev in tokenized_reviews]
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
# Print the Keywords for each of the number of topics generated by the lda_model

lda_model.print_topics()
# Create a visualization plot of your LDA

vis = gensimvis.prepare(lda_model, doc_term_matrix, dictionary)
vis
### Explain your findings from LDA and the plot above

# we can see that the most frequent words in our dataset are: flight, customer, thank, hour, service, time
# for topic 1 the top 3 frequent words in our dataset are: customer, thank, service
# for topic 2 the top 3 frequent words in our dataset are: plane, flight, crew
# and so on
# Calculate the coherence of your lda_model

from gensim.models import CoherenceModel

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenized_reviews, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score: ', coherence_lda)
# Extra: calculate the perplexity of your lda_model

print('\nPerplexity: ', lda_model.log_perplexity(doc_term_matrix)) 
#### Extra: Checking which topic is giving us the highest coherence score.
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
#stopped running it due to time limitations
model_list, coherence_values = compute_coherence_values(dictionary=dictionary,
                                                        corpus=doc_term_matrix,
                                                        texts=tokenized_reviews,
                                                        start=2,
                                                        limit=50,
                                                        step=1)
# Extra: plot the coherence values

#stopped running it due to time limitations

limit=50; start=2; step=1;
x = range(start, limit, step)

plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show() # Print the coherence scores

# Extra: Loop through the number of topics and coherence values

#stopped running it due to time limitations

for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
# Extra: Find the optimal number of topics - Justify why

# Extra: Build the optimal model and plot again the results.


# Extra: What do you observe? Discuss briefly any final findings.



### Word2Vec
Word2Vec is one of the most popular model to represent a word in a large text corpus as a vector in n-dimensional space. There are two kinds of W2V, Continuous Bag-of-Words(CBOW) and Skip-Gram. Skip-gram is used to predict the context word for a given target word. It’s reverse of CBOW algorithm. Here, target word is input while context words are output.

In most case it is known that the predictability of skip-gram is better than the one of CBOW.

We can use Word2Vec library from gensim and set the option sg which is the abbreviation of skip-gram. Use 1, if you want to set skip-gram and 0 for CBOW.
# Extra: Import the Word2Vec library

import gensim
from gensim.models import Word2Vec
# Extra: Instantiate and create a  Word2Vec() model using the tokenized_reviews.
# Optionally, set other parameters like the vector_size = 100,  window, min_count, and sg (for CBOW or skip-gram)
# Store in a variable named 'w2v'

w2v = Word2Vec(sentences=tokenized_reviews, vector_size=100, min_count=1, epochs=100) 

print(w2v)
We can find the similar words with the given word and the examples are represented below.
# Extra: using w2v, find the most similar words to the word 'crew'

w2v.wv.most_similar('crew')
# Extra: using w2v, find the most similar words to the word 'delay'

w2v.wv.most_similar('delay')

