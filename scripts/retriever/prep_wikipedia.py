
"""Here we are preprocessing the following methods for filtering and/or preparing Wikipedia docs. """

import regex as re
from html.parser import HTMLParser

PARSER = HTMLParser()
BLACKLIST = set(['23443579', '52643645']) #Here we are resolving conflicts in Wikipedia pages by using Disambiguation
#Disambiguation is a process of resolving conflicts that arise when the title of an article is ambiguous

def preprocess(article):
    #Removing HTML escaping that went uncleaned by WikiExtractor
    for k, v in article.items():
        article[k] = PARSER.unescape(v)

    # Filtering disambiguation pages escaped by WikiExtractor
    if article['id'] in BLACKLIST:
        return None
    if '(disambiguation)' in article['title'].lower():
        return None
    if '(disambiguation page)' in article['title'].lower():
        return None
    #Removing unecessary List/Index/outline pages from the documents,majorly consisting of links.
    if re.match(r'(List of .+)|(Index of .+)|(Outline of .+)',
                article['title']):
        return None

    # Returning the doc with `id` renamed to `title`
    return {'id': article['title'], 'text': article['text']}
