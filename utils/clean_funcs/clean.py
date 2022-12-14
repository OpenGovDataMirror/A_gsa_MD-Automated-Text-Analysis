from bs4 import BeautifulSoup
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from scipy import stats
import re
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from time import time
import warnings

warnings.filterwarnings('ignore')
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('stopwords')

stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 

def clean_text(doc):

    
    def strip_html_tags(text):
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text()
        return stripped_text

    def strip_urls(text):
        #url regex
        url_re = re.compile(r"""(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))""")
        stripped_text = url_re.sub('',text)
        return stripped_text

    def strip_emails(text):
        #email address regex
        email_re = re.compile(r'(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)')
        stripped_text = email_re.sub('',text)
        return stripped_text

    def strip_nonsense(text):
        # leave words that are at least three characters long, do not contain a number, and are no more 
        # than 17 chars long
        no_nonsense = re.findall(r'\b[a-z][a-z][a-z]+\b',text)
        stripped_text = ' '.join(w for w in no_nonsense if w != 'nan' and len(w) <= 17)
        return stripped_text
    
    doc = str(doc).lower()
    tag_free = strip_html_tags(doc)
    url_free = strip_urls(tag_free)
    email_free = strip_emails(url_free)
    normalized_1 = strip_nonsense(email_free)
    
    stop_free = " ".join([i for i in normalized_1.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(WordNetLemmatizer().lemmatize(word) for word in punc_free.split())
    
    return normalized

def tokenize_text(x):
    raw_text = x.tolist()

    text_data = []
    for text in raw_text:
        tokens = clean(text)
        text_data.append(tokens)
    
    return text_data

def lda_to_list (x):
    n_samples = 2000
    n_features = 1000
    n_components = 6
    n_top_words = 10
    #max_df=0.95, min_df=2,
    tf_vectorizer = CountVectorizer(
                                max_features=n_features,
                                stop_words='english',
                                   ngram_range = (1,2))

    tf = tf_vectorizer.fit_transform(x)
    
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    tf_feature_names = tf_vectorizer.get_feature_names()
    lda.fit(tf)
    temp_list =[]
    for topic_idx, topic in enumerate(lda.components_):
        #message = "Topic #%d: " % topic_idx
        message = ''
        message += ", ".join([tf_feature_names[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]])
        temp_list.append(message)
    return temp_list

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts,bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts,trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
