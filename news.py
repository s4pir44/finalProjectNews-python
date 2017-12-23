#  -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import functools
import itertools
import sys


def getSources():
    source_url = 'https://newsapi.org/v1/sources?language=en'
    response = requests.get(source_url).json()
    sources = []
    for source in response['sources']:
        sources.append(source['id'])
    print(sources)
    return sources


def mapping():
    d = {}
    response = requests.get('https://newsapi.org/v1/sources?language=en')
    response = response.json()
    for s in response['sources']:
        d[s['id']] = s['category']
    print(d)
    return d


def category(source, m):
    try:
        return m[source]
    except:
        return 'NC'


def cleanData(path):
    data = pd.read_csv(path,encoding='windows-1252')
    data = data.drop_duplicates('url')
    data.to_csv(path, index=False)


def getDailyNews():

    print(sys.getdefaultencoding())
    print(sys.getfilesystemencoding())

    sources = getSources()
    key = 'f7e6f4aa087d4fc88c82e417af753411'
    url = 'https://newsapi.org/v1/articles?source={0}&sortBy={1}&apiKey={2}'
    responses = []
    for i, source in tqdm(enumerate(sources)):
        try:
            u = url.format(source, 'top', key)
            response = requests.get(u)
            r = response.json()
            for article in r['articles']:
                article['source'] = source
            responses.append(r)
        except:
            u = url.format(source, 'latest', key)
            response = requests.get(u)
            r = response.json()
            for article in r['articles']:
                article['source'] = source
            responses.append(r)
#    elem = responses.pop(26)
#    responses.clear()
#    responses.append(elem)


#    print('responses \n')
#    print(responses)

    news = pd.DataFrame(functools.reduce(lambda x, y: x + y, map(lambda r: r['articles'], responses)))
    news = news.dropna()
    news = news.drop_duplicates()
    d = mapping()
    news['category'] = news['source'].map(lambda s: category(s, d))
    news['scraping_date'] = datetime.now()
    print('news columns \n')
    print(news.columns)
    print('news \n')
    print(news)
    try:
        aux = pd.read_csv(r'.\latest_news.csv', encoding='windows-1252')
    except:
        print("Unexpected error:", sys.exc_info()[0])
        aux = pd.DataFrame(columns=list(news.columns))
        aux.to_csv(r'.\latest_news.csv', encoding='windows-1252', index=False)

    with open(r'.\latest_news.csv', 'a', encoding='windows-1252', errors='ignore') as f:
        news.to_csv(f, header=False, encoding='windows-1252', index=False)

    cleanData(r'.\latest_news.csv')
    print('Done')


if __name__ == '__main__':
    getDailyNews()
