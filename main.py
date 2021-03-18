from pymongo import MongoClient

client = MongoClient(
    "mongodb://mongoadmin:xxxxxxxx@52.56.245.xx:xxxxx/?authSource=admin&readPreference=primary&ssl=false" --> #assume your own key to your database
)
print(client)
db = client.xxxxxx --> #assume your company name to extract data from your database. We had over 10.000 text data to process. 
print(db)
articles = db.Articles
individuals = db.Individuals


arr = []
for x in individuals.find(
    {},
    {
        "_id": 1,
        "website": 1,
        "name": 1,
        "email": 1,
        "phoneNumber": 1,
        "profilePageText": 1,
        "firm": 1,
    },
).limit(400):
    arr.append(x)

array = []
for y in articles.find({}, {"_id": 1, "website": 1, "text": 1, "firm": 1}).limit(400):
    array.append(y)

import pandas as pd

list_dataframe = pd.DataFrame(arr)
list_dataframe2 = pd.DataFrame(array)
horizontal_stack = pd.concat([list_dataframe, list_dataframe2], axis=1)
print(horizontal_stack)

import numpy as np

print(horizontal_stack["text"][0])

horizontal_stack.isnull().sum()

horizontal_stack["phoneNumber"].dropna(inplace=True, axis=0)
horizontal_stack["email"].dropna(inplace=True, axis=0)

blanks = []

for rv in horizontal_stack["text"]:
    if type(rv) == str:
        if rv.isspace():
            blanks.append(i)
print(blanks)
horizontal_stack["text"].drop(blanks, inplace=True)

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.wordnet import WordNetLemmatizer
import re
from nltk.corpus import stopwords

pattern = r"\b[^\d\W]+\b"

tokenizer = RegexpTokenizer(pattern)


def tokenizer_man(doc, remove_stopwords=False):
    doc_rem_puct = re.sub(r"[^a-zA-Z]", " ", doc)
    words = doc_rem_puct.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return words


en_stop = get_stop_words("en")
lemmatizer = WordNetLemmatizer()

stops1 = set(stopwords.words("english"))
print(stops1)

len(stopwords.words("english"))

stops1.add("newWords")
print(len(stops1))

raw = str(horizontal_stack["text"][0]).lower()
tokens = tokenizer.tokenize(raw)
" ".join(tokens)
len(tokens)

horizontal_stack["numbers"] = (
    horizontal_stack["text"].replace("[\€,]", "", regex=True).astype(str)
)
horizontal_stack["numbers"] = (
    horizontal_stack["text"].replace(",", "", regex=True).astype(str)
)
horizontal_stack["numbers"] = (
    horizontal_stack["text"].replace("[\£,]", "", regex=True).astype(str)
)
horizontal_stack["numbers"] = (
    horizontal_stack["text"].replace("[\$,]", "", regex=True).astype(str)
)

text = horizontal_stack.loc[:, "text"] = horizontal_stack.text.apply(
    lambda x: str.lower(x)
)

import re

new = horizontal_stack.loc[:, "text"] = horizontal_stack.text.apply(
    lambda x: " ".join(re.findall("[\w]+", x))
)
print(new)

from stop_words import get_stop_words

stop_words = get_stop_words("en")


def remove_stopWords(s):
    """For removing stop words"""
    s = " ".join(word for word in s.split() if word not in stop_words)
    return s


final = horizontal_stack.loc[:, "text"] = horizontal_stack.text.apply(
    lambda x: remove_stopWords(x)
)
print(final[0])


import nltk

horizontal_stack["text"] = horizontal_stack["text"].str.replace("\d+", "")
horizontal_stack["text"] = horizontal_stack["text"].str.strip()
horizontal_stack["Number_of_words"] = horizontal_stack["text"].apply(
    lambda x: len(str(x).split())
)
print(horizontal_stack.head(-5))
horizontal_stack["tokenised"] = horizontal_stack["text"].str.split()
print(horizontal_stack["tokenised"][0])

import pandas as pd
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

horizontal_stack["stemmed"] = horizontal_stack["tokenised"].apply(
    lambda x: [stemmer.stem(y) for y in x]
)
horizontal_stack = horizontal_stack.drop(columns=["tokenised"])
print(horizontal_stack["stemmed"][0])

horizontal_stack["string"] = horizontal_stack["stemmed"].apply(", ".join)
print(horizontal_stack["string"][0])

horizontal_stack["clean_string"] = horizontal_stack["string"].replace(
    ",", "", regex=True
)
print(horizontal_stack["clean_string"][0])

horizontal_stack.shape
horizontal_stack.columns

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_df=0.9, min_df=2, stop_words="english")

dtm = tfidf.fit_transform(horizontal_stack["clean_string"])
dtm


from sklearn.decomposition import NMF, LatentDirichletAllocation


nmf_model = NMF(n_components=10, random_state=42)
nmf_model.fit(dtm)


LDA = LatentDirichletAllocation(n_components=10, random_state=42)
LDA.fit(dtm)

len(tfidf.get_feature_names())

# NMF MODEL
for index, topic in enumerate(nmf_model.components_):
    print(f"THE TOP 15 WORDS FOR TOPIC #{index}")
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print("\n")

# LDA MODEL
for index, topic in enumerate(LDA.components_):
    print(f"THE TOP 15 WORDS FOR TOPIC #{index}")
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print("\n")

topic_results = nmf_model.transform(dtm)
horizontal_stack["NMF_Topic"] = topic_results.argmax(axis=1)

LDA_topic_results = LDA.transform(dtm)
horizontal_stack["LDA_Topic"] = LDA_topic_results.argmax(axis=1)

mytopic_dict = {
    0: "science",
    1: "business",
    2: "retail industry",
    3: "investment banking",
    4: "technology",
    5: "property industry",
    6: "arts and humanities",
    7: "medical industry",
    8: "constructing industry",
    9: "politics",
}

horizontal_stack["topic_label_NMF"] = horizontal_stack["NMF_Topic"].map(mytopic_dict)
horizontal_stack["topic_label_LDA"] = horizontal_stack["LDA_Topic"].map(mytopic_dict)
horizontal_stack.head(-5)


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re, nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import spacy

nlp = spacy.load("en_core_web_sm")
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.colors as mcolors
from collections import Counter
from matplotlib.ticker import FuncFormatter
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook

sns.set_style("dark")
graph = sns.catplot(
    data=horizontal_stack,
    x="NMF_Topic",
    kind="count",
    height=4.5,
    aspect=2.5,
    palette="hls",
)
graph.set_xticklabels(rotation=90)
plt.title("Frequency showing NMF topics", size=20)


plt.style.use("ggplot")
plt.figure(figsize=(12, 6))
sns.distplot(horizontal_stack["Number_of_words"], kde=False, color="red", bins=100)
plt.title("Frequency distribution of number of words for each text extracted", size=20)


tokens = horizontal_stack["text"].apply(lambda x: nltk.word_tokenize(x))


w2v_model = Word2Vec(
    tokens,
    min_count=100,
    window=10,
    size=400,
    alpha=0.03,
    min_alpha=0.0007,
    workers=4,
    seed=42,
)

one = w2v_model.wv["business"]
print(one)


two = w2v_model.wv.most_similar("treatment")
print(two)


def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(
        perplexity=50, n_components=2, init="pca", n_iter=2000, random_state=23
    )
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(15, 13))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(
            labels[i],
            xy=(x[i], y[i]),
            xytext=(5, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
        )
    plt.show()


tsne_plot(w2v_model)


dictionary = corpora.Dictionary(horizontal_stack["stemmed"])
doc_term_matrix = [dictionary.doc2bow(rev) for rev in horizontal_stack["stemmed"]]


LDA = gensim.models.ldamodel.LdaModel

# Build LDA model
lda_model = LDA(
    corpus=doc_term_matrix,
    id2word=dictionary,
    num_topics=10,
    random_state=100,
    chunksize=200,
    passes=100,
)

lda_model.print_topics()


cols = [
    color for name, color in mcolors.TABLEAU_COLORS.items()
]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(
    background_color="white",
    width=2500,
    height=1800,
    max_words=10,
    colormap="tab10",
    color_func=lambda *args, **kwargs: cols[i],
    prefer_horizontal=1.0,
)

topics = lda_model.show_topics(formatted=False)

fig, axes = plt.subplots(2, 2, figsize=(15, 15), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title("Topic " + str(i), fontdict=dict(size=16))
    plt.gca().axis("off")


def format_topics_sentences(ldamodel=None, corpus=None, texts=horizontal_stack):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 10), topic_keywords]),
                    ignore_index=True,
                )
            else:
                break
    sent_topics_df.columns = ["Group", "Perc_Contribution", "Topic_Keywords"]

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return sent_topics_df


df_topic_sents_keywords = format_topics_sentences(
    ldamodel=lda_model, corpus=doc_term_matrix, texts=horizontal_stack["stemmed"]
)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = [
    "Document_No",
    "Group",
    "Topic_Percentage_Contribution",
    "Keywords",
    "Text",
]
df_dominant_topic.head(10)


horizontal_stack["embedding_similarity1"] = w2v_model.similarity("business", "research")
print(horizontal_stack["embedding_similarity1"])

# cosine similarity

horizontal_stack["similar_words_embeddings"] = w2v_model.wv["treatment"]
print(horizontal_stack["similar_words_embeddings"])
horizontal_stack["similar_words_embeddings1"] = w2v_model.wv["healthcare"]
print(horizontal_stack["similar_words_embeddings1"])
horizontal_stack["similar_words_embeddings2"] = w2v_model.wv["research"]
print(horizontal_stack["similar_words_embeddings2"])
horizontal_stack["similar_words_embeddings3"] = w2v_model.wv["investment"]
print(horizontal_stack["similar_words_embeddings3"])
horizontal_stack["similar_words_embeddings4"] = w2v_model.wv["teams"]
print(horizontal_stack["similar_words_embeddings4"])
horizontal_stack["similar_words_embeddings5"] = w2v_model.wv["infrastructure"]
print(horizontal_stack["similar_words_embeddings5"])
horizontal_stack["similar_words_embeddings6"] = w2v_model.wv["technology"]
print(horizontal_stack["similar_words_embeddings6"])
horizontal_stack["similar_words_embeddings7"] = w2v_model.wv["customers"]
print(horizontal_stack["similar_words_embeddings7"])
horizontal_stack["similar_words_embeddings8"] = w2v_model.wv["building"]
print(horizontal_stack["similar_words_embeddings8"])
horizontal_stack["similar_words_embeddings9"] = w2v_model.wv["market"]
print(horizontal_stack["similar_words_embeddings9"])


df = pd.concat([horizontal_stack, df_dominant_topic], axis=1)
print(df)


from IPython.display import display
import pandas as pd

df = pd.DataFrame(df)

display(df)


import pandas as pd
import numpy as np
from collections import Counter
import re
import nltk
from gensim.corpora import Dictionary

from sklearn.preprocessing import LabelEncoder

from gensim.models.ldamulticore import LdaMulticore
from gensim.models.word2vec import Word2Vec
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import base64
import io
sns.set()


print(df.shape)
df.head(3)

df.isnull().sum()
df.Group.value_counts().index

fig, ax = plt.subplots(1,1,figsize=(8,6))

author_vc = df.Group.value_counts()

ax.bar(range(10), author_vc)
ax.set_xticks(range(10))
ax.set_xticklabels(author_vc.index, fontsize=16)

for rect, c, value in zip(ax.patches, ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w', 'b','r'], author_vc.values):
    rect.set_color(c)
    height = rect.get_height()
    width = rect.get_width()
    x_loc = rect.get_x()
    ax.text(x_loc + width/2, 0.9*height, value, ha='center', va='center', fontsize=18, color='white')

document_lengths = np.array(list(map(len, df.text.str.split(' '))))

print("The average number of words in a document is: {}.".format(np.mean(document_lengths)))
print("The minimum number of words in a document is: {}.".format(min(document_lengths)))
print("The maximum number of words in a document is: {}.".format(max(document_lengths)))

fig, ax = plt.subplots(figsize=(15,6))

ax.set_title("Distribution of number of words", fontsize=16)
ax.set_xlabel("Number of words")
sns.distplot(document_lengths, bins=50, ax=ax);

print("There are {} documents with over 150 words.".format(sum(document_lengths > 150)))

shorter_documents = document_lengths[document_lengths <= 150]

print("There are {} documents with tops 5 words.".format(sum(document_lengths <= 5)))

our_special_word = 'business'

def remove_ascii_words(df):
    """ removes non-ascii characters from the 'texts' column in df.
    It returns the words containig non-ascii characers.
    """
    non_ascii_words = []
    for i in range(len(df)):
        for word in df.loc[i, 'text'].split(' '):
            if any([ord(character) >= 128 for character in word]):
                non_ascii_words.append(word)
                df.loc[i, 'text'] = df.loc[i, 'text'].replace(word, our_special_word)
    return non_ascii_words

non_ascii_words = remove_ascii_words(df)

print("Replaced {} words with characters with an ordinal >= 128 in the train data.".format(
    len(non_ascii_words)))

def get_good_tokens(sentence):
    replaced_punctation = list(map(lambda token: re.sub('[^0-9A-Za-z!?]+', '', token), sentence))
    removed_punctation = list(filter(lambda token: token, replaced_punctation))
    return removed_punctation

def w2v_preprocessing(df):
    """ All the preprocessing steps for word2vec are done in this function.
    All mutations are done on the dataframe itself. So this function returns
    nothing.
    """
    df['text'] = df.text.str.lower()
    df['document_sentences'] = df.text.str.split('.')  # split texts into individual sentences
    df['tokenized_sentences'] = list(map(lambda sentences:
                                         list(map(nltk.word_tokenize, sentences)),
                                         df.document_sentences))  # tokenize sentences
    df['tokenized_sentences'] = list(map(lambda sentences:
                                         list(map(get_good_tokens, sentences)),
                                         df.tokenized_sentences))  # remove unwanted characters
    df['tokenized_sentences'] = list(map(lambda sentences:
                                         list(filter(lambda lst: lst, sentences)),
                                         df.tokenized_sentences))  # remove empty lists

w2v_preprocessing(df)

def lda_get_good_tokens(df):
    df['text'] = df.text.str.lower()
    df['tokenized_text'] = list(map(nltk.word_tokenize, df.text))
    df['tokenized_text'] = list(map(get_good_tokens, df.tokenized_text))

lda_get_good_tokens(df)

tokenized_only_dict = Counter(np.concatenate(df.tokenized_text.values))

tokenized_only_df = pd.DataFrame.from_dict(tokenized_only_dict, orient='index')
tokenized_only_df.rename(columns={0: 'count'}, inplace=True)

tokenized_only_df.sort_values('count', ascending=False, inplace=True)

# I made a function out of this since I will use it again later on 
def word_frequency_barplot(df, nr_top_words=50):
    """ df should have a column named count.
    """
    fig, ax = plt.subplots(1,1,figsize=(20,5))

    sns.barplot(list(range(nr_top_words)), df['count'].values[:nr_top_words], palette='hls', ax=ax)

    ax.set_xticks(list(range(nr_top_words)))
    ax.set_xticklabels(df.index[:nr_top_words], fontsize=14, rotation=90)
    return ax
    
ax = word_frequency_barplot(tokenized_only_df)
ax.set_title("Word Frequencies", fontsize=16);

def remove_stopwords(df):
    """ Removes stopwords based on a known set of stopwords
    available in the nltk package. In addition, we include our
    made up word in here.
    """
    # Luckily nltk already has a set of stopwords that we can remove from the texts.
    stopwords = nltk.corpus.stopwords.words('english')
    # we'll add our own special word in here 'qwerty'
    stopwords.append(our_special_word)

    df['stopwords_removed'] = list(map(lambda doc:
                                       [word for word in doc if word not in stopwords],
                                       df['tokenized_text']))

remove_stopwords(df)

def stem_words(df):
    lemm = nltk.stem.WordNetLemmatizer()
    df['lemmatized_text'] = list(map(lambda sentence:
                                     list(map(lemm.lemmatize, sentence)),
                                     df.stopwords_removed))

    p_stemmer = nltk.stem.porter.PorterStemmer()
    df['stemmed_text'] = list(map(lambda sentence:
                                  list(map(p_stemmer.stem, sentence)),
                                  df.lemmatized_text))

stem_words(df)

dictionary = Dictionary(documents=df.stemmed_text.values)

print("Found {} words.".format(len(dictionary.values())))

dictionary.filter_extremes(no_above=0.8, no_below=3)

dictionary.compactify()
print("Left with {} words.".format(len(dictionary.values())))

def document_to_bow(df):
    df['bow'] = list(map(lambda doc: dictionary.doc2bow(doc), df.stemmed_text))
    
document_to_bow(df)



def lda_preprocessing(df):
    """ All the preprocessing steps for LDA are combined in this function.
    All mutations are done on the dataframe itself. So this function returns
    nothing.
    """
    lda_get_good_tokens(df)
    remove_stopwords(df)
    stem_words(df)
    document_to_bow(df)

cleansed_words_df = pd.DataFrame.from_dict(dictionary.token2id, orient='index')
cleansed_words_df.rename(columns={0: 'id'}, inplace=True)

cleansed_words_df['count'] = list(map(lambda id_: dictionary.dfs.get(id_), cleansed_words_df.id))
del cleansed_words_df['id']

cleansed_words_df.sort_values('count', ascending=False, inplace=True)

ax = word_frequency_barplot(cleansed_words_df)
ax.set_title("Document Frequencies (Number of documents a word appears in)", fontsize=16);

import numpy as np
df['Group']=df.Group.values
print(df['Group'])
df['Group'] = df['Group'].astype(str)
print(df['Group'])


first = list(np.concatenate(df.loc[df.Group == '0.0', 'stemmed_text'].values))
second = list(np.concatenate(df.loc[df.Group == '1.0', 'stemmed_text'].values))
third = list(np.concatenate(df.loc[df.Group == '2.0', 'stemmed_text'].values))
fourth = list(np.concatenate(df.loc[df.Group == '3.0', 'stemmed_text'].values))
fifth = list(np.concatenate(df.loc[df.Group == '4.0', 'stemmed_text'].values))
sixth = list(np.concatenate(df.loc[df.Group == '5.0', 'stemmed_text'].values))
seventh = list(np.concatenate(df.loc[df.Group == '6.0', 'stemmed_text'].values))
eigth = list(np.concatenate(df.loc[df.Group == '7.0', 'stemmed_text'].values))
ninth = list(np.concatenate(df.loc[df.Group == '8.0', 'stemmed_text'].values))
tenth = list(np.concatenate(df.loc[df.Group == '9.0', 'stemmed_text'].values))

first_frequencies = {word: first.count(word) for word in cleansed_words_df.index[:50]}
second_frequencies = {word: second.count(word) for word in cleansed_words_df.index[:50]}
third_frequencies = {word: third.count(word) for word in cleansed_words_df.index[:50]}
fourth_frequencies = {word: fourth.count(word) for word in cleansed_words_df.index[:50]}
fifth_frequencies = {word: fifth.count(word) for word in cleansed_words_df.index[:50]}
sixth_frequencies = {word: sixth.count(word) for word in cleansed_words_df.index[:50]}
seventh_frequencies = {word: seventh.count(word) for word in cleansed_words_df.index[:50]}
eigth_frequencies = {word: eigth.count(word) for word in cleansed_words_df.index[:50]}
ninth_frequencies = {word: ninth.count(word) for word in cleansed_words_df.index[:50]}
tenth_frequencies = {word: tenth.count(word) for word in cleansed_words_df.index[:50]}

frequencies_df = pd.DataFrame(index=cleansed_words_df.index[:50])


frequencies_df['first_freq'] = list(map(lambda word:
                                      first_frequencies[word],
                                      frequencies_df.index))
frequencies_df['second_freq'] = list(map(lambda word:
                                          second_frequencies[word] + third_frequencies[word],
                                          frequencies_df.index))
frequencies_df['third_freq'] = list(map(lambda word:
                                              third_frequencies[word] + fourth_frequencies[word] + fifth_frequencies[word],
                                              frequencies_df.index))
frequencies_df['fourth_freq'] = list(map(lambda word:
                                      fourth_frequencies[word],
                                      frequencies_df.index))
frequencies_df['fifth_freq'] = list(map(lambda word:
                                          fifth_frequencies[word] + sixth_frequencies[word],
                                          frequencies_df.index))
frequencies_df['sixth_freq'] = list(map(lambda word:
                                              sixth_frequencies[word] + seventh_frequencies[word] + eigth_frequencies[word],
                                              frequencies_df.index))
frequencies_df['seventh_freq'] = list(map(lambda word:
                                      seventh_frequencies[word],
                                      frequencies_df.index))
frequencies_df['eigth_freq'] = list(map(lambda word:
                                          eigth_frequencies[word] + ninth_frequencies[word],
                                          frequencies_df.index))
frequencies_df['ninth_freq'] = list(map(lambda word:
                                            ninth_frequencies[word] + tenth_frequencies[word] + first_frequencies[word],
                                              frequencies_df.index))
frequencies_df['tenth_freq'] = list(map(lambda word:
                                      tenth_frequencies[word],
                                      frequencies_df.index))

fig, ax = plt.subplots(1,1,figsize=(20,5))

nr_top_words = len(frequencies_df)
nrs = list(range(nr_top_words))

sns.barplot(nrs, frequencies_df['first_freq'].values, color='b', ax=ax, label="1")
sns.barplot(nrs, frequencies_df['second_freq'].values, color='g', ax=ax, label="2")
sns.barplot(nrs, frequencies_df['third_freq'].values, color='r', ax=ax, label="3")
sns.barplot(nrs, frequencies_df['fourth_freq'].values, color='c', ax=ax, label="4")
sns.barplot(nrs, frequencies_df['fifth_freq'].values, color='w', ax=ax, label="5")
sns.barplot(nrs, frequencies_df['sixth_freq'].values, color='y', ax=ax, label="6")
sns.barplot(nrs, frequencies_df['seventh_freq'].values, color='m', ax=ax, label="7")
sns.barplot(nrs, frequencies_df['eigth_freq'].values, color='k', ax=ax, label="8")
sns.barplot(nrs, frequencies_df['ninth_freq'].values, color='r', ax=ax, label="9")
sns.barplot(nrs, frequencies_df['tenth_freq'].values, color='b', ax=ax, label="10")


ax.set_title("Word frequencies per group", fontsize=16)
ax.legend(prop={'size': 16})
ax.set_xticks(nrs)
ax.set_xticklabels(frequencies_df.index, fontsize=14, rotation=90);

corpus = df.bow

num_topics = 10
#A multicore approach to decrease training time
LDAmodel = LdaMulticore(corpus=corpus,
                        id2word=dictionary,
                        num_topics=num_topics,
                        workers=4,
                        chunksize=4000,
                        passes=7,
                        alpha='asymmetric')

def document_to_lda_features(lda_model, document):
    """ Transforms a bag of words document to features.
    It returns the proportion of how much each topic was
    present in the document.
    """
    topic_importances = LDAmodel.get_document_topics(document, minimum_probability=0)
    topic_importances = np.array(topic_importances)
    return topic_importances[:,1]

df['lda_features'] = list(map(lambda doc:
                                      document_to_lda_features(LDAmodel, doc),
                                      df.bow))


first_distribution = df.loc[df.Group == '0.0', 'lda_features'].mean()
second_distribution = df.loc[df.Group == '1.0', 'lda_features'].mean()
third_distribution = df.loc[df.Group == '2.0', 'lda_features'].mean()
fourth_distribution = df.loc[df.Group == '3.0', 'lda_features'].mean()
fifth_distribution = df.loc[df.Group == '4.0', 'lda_features'].mean()
sixth_distribution = df.loc[df.Group == '5.0', 'lda_features'].mean()
seventh_distribution = df.loc[df.Group == '6.0', 'lda_features'].mean()
eigth_distribution = df.loc[df.Group == '7.0', 'lda_features'].mean()
ninth_distribution = df.loc[df.Group == '8.0', 'lda_features'].mean()
tenth_distribution = df.loc[df.Group == '9.0', 'lda_features'].mean()
fig, [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10] = plt.subplots(10,1,figsize=(20,10))

nr_top_bars = 5

def get_topic_top_words(lda_model, topic_id, nr_top_words=5):
    """ Returns the top words for topic_id from lda_model.
    """
    id_tuples = lda_model.get_topic_terms(topic_id, topn=nr_top_words)
    word_ids = np.array(id_tuples)[:,0]
    words = map(lambda id_: lda_model.id2word[id_], word_ids)
    return words

for Group, distribution in zip(['0.0', '1.0', '2.0','3.0','4.0','5.0','6.0','7.0','8.0','9.0'], [first_distribution, second_distribution, third_distribution, fourth_distribution, fifth_distribution, sixth_distribution, seventh_distribution, eigth_distribution, ninth_distribution, tenth_distribution]):
    print("Looking up top words from top topics from {}.".format(Group))
    for x in sorted(np.argsort(distribution)[-5:]):
        top_words = get_topic_top_words(LDAmodel, x)
        print("For topic {}, the top words are: {}.".format(x, ", ".join(top_words)))
    print("")

sentences = []
for sentence_group in df.tokenized_sentences:
    sentences.extend(sentence_group)

print("Number of sentences: {}.".format(len(sentences)))
print("Number of texts: {}.".format(len(df)))

df = df.loc[:,~df.columns.duplicated()]
print(df)

num_features = 200   
min_word_count = 3   
num_workers = 4      
context = 6          
downsampling = 1e-3   


W2Vmodel = Word2Vec(sentences=sentences,
                    sg=1,
                    hs=0,
                    workers=num_workers,
                    size=num_features,
                    min_count=min_word_count,
                    window=context,
                    sample=downsampling,
                    negative=5,
                    iter=6)

def get_w2v_features(w2v_model, sentence_group):
    """ Transform a sentence_group (containing multiple lists
    of words) into a feature vector. It averages out all the
    word vectors of the sentence_group.
    """
    words = np.concatenate(sentence_group)  # words in text
    index2word_set = set(w2v_model.wv.vocab.keys())  # words known to model
    
    featureVec = np.zeros(w2v_model.vector_size, dtype="float32")
    
    # Initialize a counter for number of words in a review
    nwords = 0
    # Loop over each word in the comment and, if it is in the model's vocabulary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            featureVec = np.add(featureVec, w2v_model[word])
            nwords += 1.

    # Divide the result by the number of words to get the average
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

df['w2v_features'] = list(map(lambda sen_group:
                                      get_w2v_features(W2Vmodel, sen_group),
                                      df.tokenized_sentences))


label_encoder = LabelEncoder()

label_encoder.fit(df.Group)
df['topic_id'] = label_encoder.transform(df.Group)

def get_cross_validated_model(model, param_grid, X, y, nr_folds=5):
    """ Trains a model by doing a grid search combined with cross validation.
    args:
        model: your model
        param_grid: dict of parameter values for the grid search
    returns:
        Model trained on entire dataset with hyperparameters chosen from best results in the grid search.
    """
    # train the model (since the evaluation is based on the logloss, we'll use neg_log_loss here)
    grid_cv = GridSearchCV(model, param_grid=param_grid, scoring='neg_log_loss', cv=nr_folds, n_jobs=-1, verbose=True)
    best_model = grid_cv.fit(X, y)
    # show top models with parameter values
    result_df = pd.DataFrame(best_model.cv_results_)
    show_columns = ['mean_test_score', 'mean_train_score', 'rank_test_score']
    for col in result_df.columns:
        if col.startswith('param_'):
            show_columns.append(col)
    display(result_df[show_columns].sort_values(by='rank_test_score').head())
    return best_model

X_train_lda = np.array(list(map(np.array, df.lda_features)))
X_train_w2v = np.array(list(map(np.array, df.w2v_features)))
X_train_combined = np.append(X_train_lda, X_train_w2v, axis=1)

models = dict()

lr = LogisticRegression()

param_grid = {'penalty': ['l1', 'l2']}

best_lr_lda = get_cross_validated_model(lr, param_grid, X_train_lda, df["topic_id"])

models['best_lr_lda'] = best_lr_lda
