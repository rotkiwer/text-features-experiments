# -*- coding: utf-8 -*-

from datetime import datetime

startTime = datetime.now()

documents_train = [
    'U.S. President Barack Obama said the fight against militants in Iraq, which included new attacks today, will be a “long-term project,” tying the prospects for success to whether the nation’s leaders quickly form an inclusive government.',
    'The U.S. conducted five airstrikes against Islamic State militants today to defend Kurdish forces near Erbil, according to a statement from U.S. Central Command in Tampa, Florida. Fighter jets and armed drones destroyed several armed trucks and a mortar position held by militants, the statement said. The strikes followed four yesterday against Islamic State forces the U.S. said were attacking Yezidi civilians near Sinjar.',
    'National Collegiate Athletic Association rules barring student athletes from seeking a share of its $800 million in annual broadcast revenue are illegal, a federal judge ruled in a decision that may sweep away players’ amateur status and opens the door for them to be paid.',
    'The lawsuit, filed in 2009 by ex-college basketball player Ed O’Bannon, challenged the treatment of students as amateurs as college basketball and football evolved into multibillion-dollar businesses, with money flowing to the NCAA, broadcasters, member schools and coaches -- everyone but the players.',
]

documents_test = [
    'Obama yesterday offered no prediction for the duration of American airstrikes, while saying a government that unites Iraq’s religious and ethnic factions would allow “us to not just play defense, but also engage in some offense” against the al-Qaeda offshoot known as Islamic State.',
]

documents_train_processed = [
    [
        ('money', 10),
        ('china', 1),
        ('macau', 1),
    ],
    [
        ('london', 10),
        ('spain', 1),
        ('egg', 1),
    ]
]

documents_test_processed = [
    [
        ('money', 10),
        ('china', 1),
        ('macau', 1)
    ]
]

import numpy as np
import requests as r
import pickle
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import VectorizerMixin
from collections import Counter
from pprint import pprint


class WikiFeaturesExtractor(VectorizerMixin):

    def __init__(self):
        # TODO: take k most significant words / k% of total weight
        # TODO: take only nouns
        self.fitted = False
        self.features = []

        self.k = 10

        self.input = 'content'
        self.strip_accents = None
        self.preprocessor = None
        self.analyzer = 'word'
        self.lowercase = True
        self.stop_words = 'english'
        self.tokenizer = None
        self.token_pattern = r"(?u)\b\w\w+\b"
        self.encoding = 'utf-8'
        self.decode_error = 'strict'
        self.ngram_range = (1, 1)

        self._preprocess = self.build_analyzer()

        """
        from nltk import PorterStemmer
        stemmer = PorterStemmer()
        tokens_stemmed = [stemmer.stem_word(token) for token in tokens]
        pprint (tokens_stemmed)
        """

    # TODO: def fit

    # TODO: def get_feature_names     etc.

    """
    binary : boolean, False by default.
        If True, all non zero counts are set to 1. This is useful for discrete
        probabilistic models that model binary events rather than integer
        counts.
    """

    def fit_transform(self, documents):
        self.features = []

        documents = self._documents_prepare(documents)
        documents_as_categories = []
        documents_features_set = set()
        for document in documents:
            # self.fork_test(document)
            document_categories = self._get_document_wiki_categories(document)
            documents_as_categories.append(document_categories[0])
            documents_features_set.update(document_categories[1])

        self.features = sorted(list(documents_features_set))
        self.fitted = True
        pprint(self.features)

        return self._get_transformed_for_documents_as_categories(documents_as_categories)

    def transform(self, documents):
        if not self.fitted:
            raise Exception('Extractor not fitted yet')

        documents = self._documents_prepare(documents)
        documents_as_categories = []
        for document in documents:
            documents_as_categories.append(self._get_document_wiki_categories(document)[0])

        return self._get_transformed_for_documents_as_categories(documents_as_categories)

    def _documents_prepare(self, documents):
        documents_prepared = []
        for document in documents:
            document = self._preprocess(document)
            documents_prepared.append(Counter(document).most_common(self.k))
        return documents_prepared

    def _get_transformed_for_documents_as_categories(self, documents_as_categories):
        if not self.fitted:
            raise Exception('Extractor not fitted yet')

        transformed_row, transformed_col, transformed_val = [], [], []
        doc_i = 0
        for document in documents_as_categories:
            for category_wrapper in document:
                category, weight = category_wrapper[0], category_wrapper[1]
                feature_i = np.searchsorted(self.features, category)
                transformed_row.append(doc_i)
                transformed_col.append(feature_i)
                transformed_val.append(weight)
            doc_i += 1

        return csr_matrix((transformed_val, (transformed_row, transformed_col)),
                          shape=(len(documents_as_categories), len(self.features)))

    def _get_wiki_categories_for_word(self, word):

        WIKIPEDIA_API_URL = "http://en.wikipedia.org/w/api.php?action=query&prop=categories&titles="
        WIKIPEDIA_API_PARAMS = "&cllimit=500&clshow=!hidden&format=json"

        # word should not contain ' ' or '&' but just in case
        def prepare_string_for_url(c):
            return c.replace(" ", "%20").replace("&", "%26")

        url = WIKIPEDIA_API_URL + prepare_string_for_url(word).capitalize() + WIKIPEDIA_API_PARAMS

        try:
            response = r.request('GET', url, timeout=5.0)
        except (r.exceptions.ConnectionError, r.exceptions.Timeout):
            import time
            time.sleep(2)
            return []
        print "Status: [%s] URL: %s" % (response.status_code, url)
        json = response.json()
        categories_wrapper = json['query']['pages'].values()[0]
        categories_list = []
        if 'categories' in categories_wrapper:
            # remove the prefix 'Category:'
            categories_list = [category['title'][9:] for category in categories_wrapper['categories']
                               if category['title'] not in ('Category:Disambiguation pages',
                                                            'Category:Disambiguation pages with given-name-holder lists')]
        return categories_list

    def _get_document_wiki_categories(self, document):
        document_categories = []
        document_categories_set = set()
        for word in document:
            word_string, word_weight = word[0], word[1]
            # TODO: fork
            word_categories = self._get_wiki_categories_for_word(word_string)
            for word_category in word_categories:
                if word_category not in document_categories_set:
                    document_categories_set |= {word_category}
                    document_categories.append((word_category, word_weight))
                    # document_categories.append((word_category, 1))
                else:
                    category_index = [c[0] for c in document_categories].index(word_category)
                    document_categories[category_index] = (document_categories[category_index][0],
                                                           document_categories[category_index][1] + word_weight)
                    # document_categories[category_index] = (document_categories[category_index][0],
                    #                                       1)
        return document_categories, document_categories_set

    def fork_test(self, words):

        import gevent.monkey
        gevent.monkey.patch_socket()
        from gevent.pool import Pool
        import requests

        def prepare_string_for_url(c):
            return c.replace(" ", "%20").replace("&", "%26")

        def fetch(word):
            WIKIPEDIA_API_URL = "http://en.wikipedia.org/w/api.php?action=query&prop=categories&titles="
            WIKIPEDIA_API_PARAMS = "&cllimit=500&clshow=!hidden&format=json"
            url = WIKIPEDIA_API_URL + prepare_string_for_url(word).capitalize() + WIKIPEDIA_API_PARAMS
            response = requests.request('GET', url, timeout=30.0)
            print "Status: [%s] URL: %s" % (response.status_code, url)
            json = response.json()
            categories_wrapper = json['query']['pages'].values()[0]
            """
            if 'categories' in categories_wrapper:
                for category in categories_wrapper['categories']:
                    #print category['title']
            else:
                print 'No categories found'
            """

        pool = Pool(100)
        for word in words:
            pool.spawn(fetch, word[0])
        pool.join()


__extractor_trained = True
__extractor_train_pickle_file = 'extractor_train.data'
__extractor_tested = True
__extractor_test_pickle_file = 'extractor_test.data'

from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories)
print len(newsgroups_train.data)
newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories)
print len(newsgroups_test.data)

if not __extractor_trained:
    print 'Did not load train extractor data. Calculating...'

    extractor = WikiFeaturesExtractor()
    documents_train_transformed = extractor.fit_transform(newsgroups_train.data[:100])
    from cloud.serialization.cloudpickle import dump

    with open(__extractor_train_pickle_file, 'wb') as f:
        dump((extractor, documents_train_transformed), f)

else:
    print 'Train extractor data loaded from file'

    with open(__extractor_train_pickle_file, 'rb') as f:
        extractor, documents_train_transformed = pickle.load(f)

if not __extractor_tested:
    print 'Did not load test extractor data. Calculating...'

    documents_test_transformed = extractor.transform(newsgroups_test.data[:10])
    from cloud.serialization.cloudpickle import dump

    with open(__extractor_test_pickle_file, 'wb') as f:
        dump(documents_test_transformed, f)

else:
    print 'Test extractor data loaded from file'

    with open(__extractor_test_pickle_file, 'rb') as f:
        documents_test_transformed = pickle.load(f)

print documents_train_transformed
print documents_test_transformed

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

clf = MultinomialNB(alpha=.01)
clf.fit(documents_train_transformed, newsgroups_train.target[:100])

pred = clf.predict(documents_test_transformed)
print metrics.f1_score(newsgroups_test.target[:10], pred)

print datetime.now() - startTime
