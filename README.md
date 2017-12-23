# finalProjectNews
Download Anaconda from https://www.anaconda.com/
follow instructions in https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/ 
This will:
a.Install Anaconda
b.Start and Update Anaconda
c.Update scikit-learn Library
d.Install Deep Learning Libraries


Now install the following missing libraries
first run: conda update pip
tqdm (a progress bar python utility) from this command: pip install tqdm
nltk (for natural language processing) from this command: conda install -c anaconda nltk=3.2.2
bokeh (for interactive data viz) from this command: conda install bokeh
lda (the python implementation of Latent Dirichlet Allocation) from this command: pip install lda
pyldavis (python package to visualize lda topics): pip install pyldavis


Now run the ntlkCorpusDownload.py to download corpus of ntlk for stop words and tokenization words
After all the corpus of all the packages will be installed you can run the news mining programs

Your server will have to run 2 files each 5 minutes: first news.py and then newsKnnModule.py.

first news.py will take all the latest news articles by doing scraping from the news api site and store them in latest_news.csv file
For each article we scraped we'll collect these fields:

author
title
description
url
urlToImage
publishedAt
And add two other features:

category
scraping_date : the time at which the script runs. This will help us track the data.

Then you can run the main engine of the news clustering system 
By running the following file: newsKnnModule.py

This will enable us to

- have a look at the dataset and inspect it
- apply some preoprocessings on the texts: tokenization, tf-idf
- cluster the articles using two different algorithms (Kmeans and LDA)
- visualize the clusters using Bokeh and pyldavis


At the end I have created the final csv file lda_df.csv that contains the following fields that can be used by the gui
description	category	url	topic

similar articles will get the same topic number
each article has its url to the full article, short description and the category it is classified to.

I also added some visual graphs in html files that contain the clustering map for all of the articles that were classified by running my main engine file


