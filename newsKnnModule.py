# pandas for data manipulation
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
# nltk for nlp
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
# list of stopwords like articles, preposition
stop = set(stopwords.words('english'))
from string import punctuation
from collections import Counter
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_file, output_notebook
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import lda
from sklearn.feature_extraction.text import CountVectorizer
import logging
logging.getLogger("lda").setLevel(logging.WARNING)
from bokeh.models import ColumnDataSource
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
import pyLDAvis

def keywords(category):
    tokens = data[data['category'] == category]['tokens']
    alltokens = []
    for token_list in tokens:
        alltokens += token_list
    counter = Counter(alltokens)
    return counter.most_common(10)


def tokenizer(text):
    try:
        tokens_ = [word_tokenize(sent) for sent in sent_tokenize(text)]

        tokens = []
        for token_by_sent in tokens_:
            tokens += token_by_sent

        tokens = list(filter(lambda t: t.lower() not in stop, tokens))
        tokens = list(filter(lambda t: t not in punctuation, tokens))
        tokens = list(filter(lambda t: t not in [u"'s", u"n't", u"...", u"''", u'``',
                                                 u'\u2014', u'\u2026', u'\u2013'], tokens))
        filtered_tokens = []
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)

        filtered_tokens = list(map(lambda token: token.lower(), filtered_tokens))

        return filtered_tokens
    except:
        print('Error')


def prepareLDAData():
    data = {
        'vocab': vocab,
        'doc_topic_dists': lda_model.doc_topic_,
        'doc_lengths': list(lda_df['len_docs']),
        'term_frequency':cvectorizer.vocabulary_,
        'topic_term_dists': lda_model.components_
    }
    return data


if __name__ == '__main__':
    data = pd.read_csv(r'.\news.csv', encoding='utf-8')
    #data = pd.read_csv(r'.\latest_news.csv', encoding='windows-1252')
    data.head()
    print('data shape:', data.shape)
    #data.category.value_counts().plot(kind='bar', grid=True, figsize=(16, 9))
    #plt.show()
    # remove duplicate description columns
    data = data.drop_duplicates('description')
    # remove rows with empty descriptions
    data = data[~data['description'].isnull()]
    print('data shape:', data.shape)
    data['len'] = data['description'].map(len)
    data = data[data.len > 140]
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    print('data shape:', data.shape)

    data['tokens'] = data['description'].map(tokenizer)
    for descripition, tokens in zip(data['description'].head(5), data['tokens'].head(5)):
        print('description:', descripition)
        print('tokens:', tokens)
        print()

    for category in set(data['category']):
        print('category :', category)
        print('top 10 keywords:', keywords(category))
        print('---')



    # min_df is minimum number of documents that contain a term t
    # max_features is maximum number of unique tokens (across documents) that we'd consider
    # TfidfVectorizer preprocesses the descriptions using the tokenizer we defined above

    vectorizer = TfidfVectorizer(min_df=10, max_features=10000, tokenizer=tokenizer, ngram_range=(1, 2))
    vz = vectorizer.fit_transform(list(data['description']))
    print(vz.shape)

    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    tfidf = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidf), orient='index')
    tfidf.columns = ['tfidf']

    ##############################################
    tfidf.tfidf.hist(bins=50, figsize=(15,7))
    #plt.hist(tfidf)
    #plt.show(
    ############################################

    print(tfidf.sort_values(by=['tfidf'], ascending=True).head(30))
    print('***************************************')
    print(tfidf.sort_values(by=['tfidf'], ascending=False).head(30))
    print('$$$$$$$$$$$$$$$$$$$$$$$')
    print(vz.shape[1])


    svd = TruncatedSVD(n_components=min(50, vz.shape[1]-1), random_state=0)
    svd_tfidf = svd.fit_transform(vz)
    #print(vz.shape)
    print(svd_tfidf.shape)

    tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
    tsne_tfidf = tsne_model.fit_transform(svd_tfidf)
    print(tsne_tfidf.shape)

    #plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="tf-idf clustering of the news", tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave", x_axis_type=None, y_axis_type=None, min_border=1)
    #tfidf_df = pd.DataFrame(tsne_tfidf, columns=['x', 'y'])
    #tfidf_df['description'] = data['description']
    #tfidf_df['category'] = data['category']
    #plot_tfidf.scatter(x='x', y='y', source=tfidf_df)
    #hover = plot_tfidf.select(dict(type=HoverTool))
    #hover.tooltips={"description": "@description", "category":"@category"}
    #show(plot_tfidf)


    num_clusters = 30
    kmeans_model = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1,init_size=1000, batch_size=1000, verbose=False, max_iter=1000)
    kmeans = kmeans_model.fit(vz)
    kmeans_clusters = kmeans.predict(vz)
    kmeans_distances = kmeans.transform(vz)

    for (i, desc),category in zip(enumerate(data.description),data['category']):
        if(i < 5):
            print("Cluster " + str(kmeans_clusters[i]) + ": " + desc +
                  "(distance: " + str(kmeans_distances[i][kmeans_clusters[i]]) + ")")
            print('category: ',category)
            print('---')

    sorted_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(num_clusters):
        print("Cluster %d:" % i)
        aux = ''
        for j in sorted_centroids[i, :10]:
            aux += terms[j] + ' | '
        print(aux)
        print()

    tsne_kmeans = tsne_model.fit_transform(kmeans_distances)

    colormap = np.array(["#6d8dca", "#69de53", "#723bca", "#c3e14c", "#c84dc9", "#68af4e", "#6e6cd5","#e3be38",
                         "#4e2d7c", "#5fdfa8", "#d34690", "#3f6d31", "#d44427", "#7fcdd8", "#cb4053", "#5e9981",
                         "#803a62", "#9b9e39", "#c88cca", "#e1c37b", "#34223b", "#bdd8a3", "#6e3326", "#cfbdce",
                         "#d07d3c","#52697d", "#7d6d33", "#d27c88", "#36422b", "#b68f79"])

    plot_kmeans = bp.figure(plot_width=700, plot_height=600, title="KMeans clustering of the news", tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave", x_axis_type=None, y_axis_type=None, min_border=1)
    kmeans_df = pd.DataFrame(tsne_kmeans, columns=['x', 'y'])
    kmeans_df['cluster'] = kmeans_clusters
    kmeans_df['description'] = data['description']
    kmeans_df['category'] = data['category']
    kmeans_df['url'] = data['url']

    colorList = []
    for i in kmeans_clusters:
        colorList.append(colormap[i])

    kmeans_df['color'] = colorList
    kmeans_df.to_csv('.\kmeansdf.csv', index=False)
    #import matplotlib as mpl



    plot_kmeans.scatter(x='x', y='y',
                        color='color',
                        source=kmeans_df)
    hover = plot_kmeans.select(dict(type=HoverTool))
    hover.tooltips = {"description": "@description", "category": "@category", "cluster": "@cluster"}
#    show(plot_kmeans)
    output_file(".\kmeans_scatter.html", title="kmeans_scatter")
    show(plot_kmeans)

    N = 4000
    x = np.random.random(size=N) * 100
    y = np.random.random(size=N) * 100
    radii = np.random.random(size=N) * 1.5
    colors = [
        "#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50 + 2 * x, 30 + 2 * y)
    ]
    #kmeans_df['color'] = colors[kmeans_clusters]
    # TOOLS = "hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"

    #p = figure(tools=TOOLS)

    sourcekmeans = ColumnDataSource(data=dict(x=kmeans_df['x'], y=kmeans_df['y']))

    #sourcekmeans = ColumnDataSource(data=dict(x=kmeans_df['x'], y=kmeans_df['y'],description=kmeans_df['description'], category=kmeans_df['category'], cluster=kmeans_df['cluster']))
    # plot_kmeans.circle(x='x', y='y', fill_color=colors, fill_alpha=0.6,line_color=None, source=sourcekmeans)
    # plot_kmeans.scatter(x='x', y='y', radius=radii,
    #          fill_color=colors, fill_alpha=0.6,
    #          line_color=None, source=sourcekmeans)
    #hover = plot_kmeans.select(dict(type=HoverTool))
    #hover.tooltips = {"description": "@description", "category": "@category", "cluster": "@cluster"}
    #show(plot_kmeans)

    #convert_first_to_generator = (str(w) for w in kmeans_clusters)
    #convert_first_to_generator2 = (np.array([colormap[w]]) for w in kmeans_clusters)
    convert_first_to_generator3 = (str(colormap[w]) for w in kmeans_clusters)
    #sourceColors = ColumnDataSource(data=dict(zip(convert_first_to_generator, convert_first_to_generator2)))
    #plot_kmeans.scatter(x='x', y='y', radius=radii,
    #      fill_color=colors, fill_alpha=0.6,
    #      line_color=None, source=ColumnDataSource(kmeans_df))
    #hover = plot_kmeans.select(dict(type=HoverTool))
    #hover.tooltips={"convert_first_to_generator3description": "@description", "category": "@category", "cluster":"@cluster"}
    #show(plot_kmeans)

    # _____________________________
    #plot_kmeans.circle(x=kmeans_df['x'], y=kmeans_df['y'], fill_color=colors, fill_alpha=0.6,line_color=None)
    #hover = plot_kmeans.select(dict(type=HoverTool))
    #hover.tooltips = {"description": "@description", "category": "@category", "cluster": "@cluster"}
    #show(plot_kmeans)
    # _____________________________

    cvectorizer = CountVectorizer(min_df=4, max_features=10000, tokenizer=tokenizer, ngram_range=(1,2))
    cvz = cvectorizer.fit_transform(data['description'])

    n_topics = 20
    n_iter = 2000
    lda_model = lda.LDA(n_topics=n_topics, n_iter=n_iter)
    X_topics = lda_model.fit_transform(cvz)
    n_top_words = 8
    topic_summaries = []

    topic_word = lda_model.topic_word_  # get the topic words
    vocab = cvectorizer.get_feature_names()
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        topic_summaries.append(' '.join(topic_words))
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))

    print('________')

    tsne_lda = tsne_model.fit_transform(X_topics)

    doc_topic = lda_model.doc_topic_
    lda_keys = []
    for i, tweet in enumerate(data['description']):
        lda_keys += [doc_topic[i].argmax()]

    plot_lda = bp.figure(plot_width=700, plot_height=600, title="LDA topic visualization",
        tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
        x_axis_type=None, y_axis_type=None, min_border=1)

    lda_df = pd.DataFrame(tsne_lda, columns=['x','y'])
    lda_df['description'] = data['description']
    lda_df['category'] = data['category']
    lda_df['url'] = data['url']

    lda_df['topic'] = lda_keys
    lda_df['topic'] = lda_df['topic'].map(int)

    colorList = []
    for i in lda_keys:
        colorList.append(colormap[i])

    lda_df['color'] = colorList
    plot_lda.scatter(source=lda_df, x='x', y='y', color='color')

    hover = plot_lda.select(dict(type=HoverTool))
    hover.tooltips = {"description": "@description", "topic": "@topic", "category": "@category"}
    output_file(".\plot_lda_scatter.html", title="plot_lda_scatter")
    show(plot_lda)

    lda_df.to_csv('.\\lda_df.csv', index=False)
    lda_df['len_docs'] = data['tokens'].map(len)

    ldadata = prepareLDAData()

    prepared_data = pyLDAvis.prepare(**ldadata)
    pyLDAvis.save_html(prepared_data,r'.\pyldadavis.html')