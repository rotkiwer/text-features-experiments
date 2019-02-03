# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import reuters
from collections import namedtuple
import re
import numpy as np
from datetime import datetime
import warnings
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.feature_selection import SelectPercentile, SelectKBest, VarianceThreshold, chi2, f_classif
from sklearn.random_projection import SparseRandomProjection
import matplotlib.pyplot as plt
from gensim import models, matutils
import pickle
from cloud.serialization.cloudpickle import dump


def get_data(subset, corpus='20newsgroups'):
    if 'reuters' == corpus:
        # tylko kategorie z wiecej niz 100 obiektami
        categories = [category for category in reuters.categories() if len(reuters.fileids(category)) > 100]
        # categories = ['acq', 'bop', 'coffee', 'corn', 'crude', 'dlr']

        if 'train' == subset:
            pattern = 'training'
        else:
            pattern = 'test'

        data = [(fileid, reuters.categories(fileid)) for fileid in reuters.fileids()]
        subset_getter = re.compile(pattern + '.*')
        # dokumenty z danego podzbioru i przypisane tylko do jednej kategorii
        data = [(reuters.raw(document[0]), document[1][0]) for document in data
                if subset_getter.match(document[0]) and len(document[1]) == 1]
        data = [document for document in data if document[1] in categories]

        data = np.array(data)
        Result = namedtuple('result', 'data target')
        return Result(data[:, 0], data[:, 1])

    else:
        # categories = ['alt.atheism', 'comp.graphics']
        # categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
        categories = None  # means all
        remove = ['headers', 'footers', 'quotes']
        # remove = []

        # return fetch_20newsgroups(subset=subset, categories=categories, shuffle=True, random_state=42, remove=remove)
        return fetch_20newsgroups(subset=subset, categories=categories, remove=remove)


def time_it(func, *args):
    start_time = datetime.now()
    result = func(*args)
    time_diff = datetime.now() - start_time
    return result, time_diff


def train_and_save_selector(name, corpus, extractor, selector):
    data_train = get_data('train', corpus)
    data_train_transformed = extractor.fit_transform(data_train.data)
    x_none_return, sel_train_time = time_it(selector.fit, data_train_transformed, data_train.target)
    with open(name + '.selector', 'wb') as f:
        dump((name, sel_train_time, selector), f)
    print name, 'fitted and saved in:', sel_train_time


def train_and_save_selectors(combinations):
    for name, corpus, extractor, selector_sequence in combinations:
        for selector_name, selector_object in selector_sequence:
            train_and_save_selector(name + '_' + selector_name, corpus, extractor, selector_object)


def read_selector(name):
    with open(name + '.selector', 'rb') as f:
        name_saved, sel_train_time, selector = pickle.load(f)
    return selector


class TopicExtractor:

    def __init__(self, n_topics, model='lsi'):
        self.model_type = model
        self.model = None
        self.n_topics = n_topics

    # its unsupervised learning, keeping 'target' to keep interface compatibility
    def fit(self, data, target=None):
        data = matutils.Sparse2Corpus(data, documents_columns=False)
        if self.model_type == 'lsi':
            self.model = models.LsiModel(data, self.n_topics)
        else:
            self.model = models.LdaModel(data, self.n_topics)

    def transform(self, data):
        data = matutils.Sparse2Corpus(data, documents_columns=False)
        return matutils.corpus2csc(self.model[data]).T


class TopicExtractorWithChi2:

    def __init__(self, n_topics, model='lsi'):
        self.model_type = model
        self.model = None
        self.n_topics = n_topics
        self.selector = SelectPercentile(chi2, 10)

    # its unsupervised learning, keeping 'target' to keep interface compatibility
    def fit(self, data, target=None):
        data = self.selector.fit_transform(data, target)
        data = matutils.Sparse2Corpus(data, documents_columns=False)
        if self.model_type == 'lsi':
            self.model = models.LsiModel(data, self.n_topics)
        else:
            self.model = models.LdaModel(data, self.n_topics)

    # its unsupervised learning, keeping 'target' to keep interface compatibility
    def fit_transform(self, data, target=None):
        data = self.selector.fit_transform(data, target)
        data = matutils.Sparse2Corpus(data, documents_columns=False)
        if self.model_type == 'lsi':
            self.model = models.LsiModel(data, self.n_topics)
        else:
            self.model = models.LdaModel(data, self.n_topics)
        return matutils.corpus2csc(self.model[data]).T

    def transform(self, data):
        data = self.selector.transform(data)
        data = matutils.Sparse2Corpus(data, documents_columns=False)
        return matutils.corpus2csc(self.model[data]).T


def test_combination(corpus, classifier, extractor, selector=None, selector_fitted=False):
    # TRAIN
    data_train = get_data('train', corpus)
    data_train_transformed, ext_train_time = time_it(extractor.fit_transform, (data_train.data))
    ext_feature_number = data_train_transformed.shape[1]
    if selector is not None:
        if not selector_fitted:
            data_train_transformed, sel_train_time = time_it(selector.fit_transform, data_train_transformed,
                                                             data_train.target)
        else:
            data_train_transformed = selector.transform(data_train_transformed)
            sel_train_time = 0
    else:
        sel_train_time = 0
    sel_feature_number = data_train_transformed.shape[1]
    clf_fit_time = time_it(classifier.fit, data_train_transformed, data_train.target)[1]

    # TEST
    data_test = get_data('test', corpus)
    data_test_transformed, ext_test_time = time_it(extractor.transform, data_test.data)
    if selector is not None:
        data_test_transformed, sel_test_time = time_it(selector.transform, data_test_transformed)
    else:
        sel_test_time = 0
    prediction, clf_pred_time = time_it(classifier.predict, data_test_transformed)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f1 = metrics.f1_score(data_test.target, prediction, average='macro')
        precision = metrics.precision_score(data_test.target, prediction, average='macro')
        recall = metrics.recall_score(data_test.target, prediction, average='macro')
        accuracy = metrics.accuracy_score(data_test.target, prediction)
        train_error = classifier.score(data_train_transformed, data_train.target)
        test_error = classifier.score(data_test_transformed, data_test.target)

        # CONFUSION MATRIX

        # from sklearn.metrics import confusion_matrix
        #
        # import matplotlib.pyplot as plt
        #
        # # Compute confusion matrix
        # cm = confusion_matrix(data_test.target, prediction)
        #
        # print(cm)
        #
        # # Show confusion matrix in a separate window
        # plt.matshow(cm)
        # plt.title('')
        # # plt.matshow(data, cmap=plt.cm.Blues)
        # # http://scikit-learn.org/stable/auto_examples/bicluster/plot_spectral_coclustering.html#example-bicluster-plot-spectral-coclustering-py
        # plt.colorbar()
        # plt.ylabel('Oczekiwana klasa')
        # plt.xlabel('Przypisana klasa')
        # plt.show()

    result = {
        'f1': f1, 'precision': precision, 'recall': recall, 'accuracy': accuracy,
        'train_error': train_error, 'test_error': test_error,
        'ext_feature_number': ext_feature_number, 'sel_feature_number': sel_feature_number,
        'ext_train_time': ext_train_time, 'sel_train_time': sel_train_time, 'clf_fit_time': clf_fit_time,
        'ext_test_time': ext_test_time, 'sel_test_time': sel_test_time, 'clf_pred_time': clf_pred_time
    }
    return result


def prep_result(corpus, classifier, extractor, selector, result):
    sep = '|'
    result_str = corpus + sep + classifier + sep + extractor + sep
    if selector is not None:
        result_str += selector + sep
    else:
        result_str += '-' + sep
    result_str += str(result['f1']) + sep + str(result['precision']) + sep + str(result['recall']) + sep \
                  + str(result['accuracy']) + sep + str(result['train_error']) + sep + str(result['test_error']) + sep \
                  + str(result['ext_feature_number']) + sep + str(result['sel_feature_number']) + sep \
                  + str(result['ext_train_time']) + sep + str(result['sel_train_time']) + sep \
                  + str(result['clf_fit_time']) + sep + str(result['ext_test_time']) + sep \
                  + str(result['sel_test_time']) + sep + str(result['clf_pred_time']) + sep
    return result_str


def test_selectors(classification_pipe, selector_sequences):
    for corpus, clf, ext in classification_pipe:
        feature_number_pipe = []
        f1_pipe = []
        selector_name_pipe = []

        for selector_name, selector_sequence in selector_sequences:
            f1_sequence = []
            feature_number_sequence = []
            for sel in selector_sequence:
                result = test_combination(corpus, clf[1], ext[1], sel[1], sel[2])
                feature_number_sequence.append(result['sel_feature_number'])
                f1_sequence.append(result['f1'])
                print prep_result(corpus, clf[0], ext[0], sel[0], result)
            f1_pipe.append(f1_sequence)
            feature_number_pipe.append(feature_number_sequence)
            selector_name_pipe.append(selector_name)

        plt.figure()
        plt.xlabel('Liczba wyselekcjonowanych cech')
        plt.ylabel('F1')
        for feature_number_series, f1_series, selector_name in zip(feature_number_pipe, f1_pipe, selector_name_pipe):
            plt.plot(feature_number_series, f1_series, label=selector_name)
            plt.legend(loc='best')
        # plt.show()
        plt.savefig(corpus + '_' + clf[0] + '.png')


chi2_selectors = [
    ('chi2-select-10', SelectKBest(chi2, 10), False),
    ('chi2-select-50', SelectKBest(chi2, 50), False),
    ('chi2-select-100', SelectKBest(chi2, 100), False),
    ('chi2-select-200', SelectKBest(chi2, 200), False),
    ('chi2-select-300', SelectKBest(chi2, 300), False),
    ('chi2-select-400', SelectKBest(chi2, 400), False),
    ('chi2-select-500', SelectKBest(chi2, 500), False),
    ('chi2-select-600', SelectKBest(chi2, 600), False),
    ('chi2-select-700', SelectKBest(chi2, 700), False),
    ('chi2-select-800', SelectKBest(chi2, 800), False),
    ('chi2-select-900', SelectKBest(chi2, 900), False),
    ('chi2-select-1000', SelectKBest(chi2, 1000), False),
    # ('chi2-select-all', None)
]

random_projection_transforms_20ng_t2_fitted = [
    ('random-projection-10', read_selector('20ng_t2_random-projection-10'), True),
    ('random-projection-50', read_selector('20ng_t2_random-projection-50'), True),
    ('random-projection-100', read_selector('20ng_t2_random-projection-100'), True),
    ('random-projection-200', read_selector('20ng_t2_random-projection-200'), True),
    ('random-projection-300', read_selector('20ng_t2_random-projection-300'), True),
    ('random-projection-400', read_selector('20ng_t2_random-projection-400'), True),
    ('random-projection-500', read_selector('20ng_t2_random-projection-500'), True),
    ('random-projection-600', read_selector('20ng_t2_random-projection-600'), True),
    ('random-projection-700', read_selector('20ng_t2_random-projection-700'), True),
    ('random-projection-800', read_selector('20ng_t2_random-projection-800'), True),
    ('random-projection-900', read_selector('20ng_t2_random-projection-900'), True),
    ('random-projection-1000', read_selector('20ng_t2_random-projection-1000'), True),
    # ('random-projection-none', None)
]

random_projection_transforms_reuters_t2_fitted = [
    ('random-projection-10', read_selector('reuters_t2_random-projection-10'), True),
    ('random-projection-50', read_selector('reuters_t2_random-projection-50'), True),
    ('random-projection-100', read_selector('reuters_t2_random-projection-100'), True),
    ('random-projection-200', read_selector('reuters_t2_random-projection-200'), True),
    ('random-projection-300', read_selector('reuters_t2_random-projection-300'), True),
    ('random-projection-400', read_selector('reuters_t2_random-projection-400'), True),
    ('random-projection-500', read_selector('reuters_t2_random-projection-500'), True),
    ('random-projection-600', read_selector('reuters_t2_random-projection-600'), True),
    ('random-projection-700', read_selector('reuters_t2_random-projection-700'), True),
    ('random-projection-800', read_selector('reuters_t2_random-projection-800'), True),
    ('random-projection-900', read_selector('reuters_t2_random-projection-900'), True),
    ('random-projection-1000', read_selector('reuters_t2_random-projection-1000'), True),
    # ('random-projection-none', None)
]

chi2_lsi_transforms_20ng_t2_fitted = [
    ('chi2+lsi-10', read_selector('20ng_t2_chi2+lsi-10'), True),
    ('chi2+lsi-50', read_selector('20ng_t2_chi2+lsi-50'), True),
    ('chi2+lsi-100', read_selector('20ng_t2_chi2+lsi-100'), True),
    ('chi2+lsi-200', read_selector('20ng_t2_chi2+lsi-200'), True),
    ('chi2+lsi-300', read_selector('20ng_t2_chi2+lsi-300'), True),
    ('chi2+lsi-400', read_selector('20ng_t2_chi2+lsi-400'), True),
    ('chi2+lsi-500', read_selector('20ng_t2_chi2+lsi-500'), True),
    ('chi2+lsi-600', read_selector('20ng_t2_chi2+lsi-600'), True),
    ('chi2+lsi-700', read_selector('20ng_t2_chi2+lsi-700'), True),
    ('chi2+lsi-800', read_selector('20ng_t2_chi2+lsi-800'), True),
    ('chi2+lsi-900', read_selector('20ng_t2_chi2+lsi-900'), True),
    ('chi2+lsi-1000', read_selector('20ng_t2_chi2+lsi-1000'), True),
    # ('lsi-none', None)
]

chi2_lsi_transforms_reuters_t2_fitted = [
    ('chi2+lsi-10', read_selector('reuters_t2_chi2+lsi-10'), True),
    ('chi2+lsi-50', read_selector('reuters_t2_chi2+lsi-50'), True),
    ('chi2+lsi-100', read_selector('reuters_t2_chi2+lsi-100'), True),
    ('chi2+lsi-200', read_selector('reuters_t2_chi2+lsi-200'), True),
    ('chi2+lsi-300', read_selector('reuters_t2_chi2+lsi-300'), True),
    ('chi2+lsi-400', read_selector('reuters_t2_chi2+lsi-400'), True),
    ('chi2+lsi-500', read_selector('reuters_t2_chi2+lsi-500'), True),
    ('chi2+lsi-600', read_selector('reuters_t2_chi2+lsi-600'), True),
    ('chi2+lsi-700', read_selector('reuters_t2_chi2+lsi-700'), True),
    ('chi2+lsi-800', read_selector('reuters_t2_chi2+lsi-800'), True),
    ('chi2+lsi-900', read_selector('reuters_t2_chi2+lsi-900'), True),
    ('chi2+lsi-1000', read_selector('reuters_t2_chi2+lsi-1000'), True),
    # ('lsi-none', None)
]

chi2_lda_transforms_20ng_t2_fitted = [
    ('chi2+lda-10', read_selector('20ng_t2_chi2+lda-10'), True),
    ('chi2+lda-50', read_selector('20ng_t2_chi2+lda-50'), True),
    ('chi2+lda-100', read_selector('20ng_t2_chi2+lda-100'), True),
    ('chi2+lda-200', read_selector('20ng_t2_chi2+lda-200'), True),
    ('chi2+lda-300', read_selector('20ng_t2_chi2+lda-300'), True),
    ('chi2+lda-400', read_selector('20ng_t2_chi2+lda-400'), True),
    ('chi2+lda-500', read_selector('20ng_t2_chi2+lda-500'), True),
    ('chi2+lda-600', read_selector('20ng_t2_chi2+lda-600'), True),
    ('chi2+lda-700', read_selector('20ng_t2_chi2+lda-700'), True),
    ('chi2+lda-800', read_selector('20ng_t2_chi2+lda-800'), True),
    ('chi2+lda-900', read_selector('20ng_t2_chi2+lda-900'), True),
    ('chi2+lda-1000', read_selector('20ng_t2_chi2+lda-1000'), True),
    # ('lda-none', None)
]

chi2_lda_transforms_reuters_t2_fitted = [
    ('chi2+lda-10', read_selector('reuters_t2_chi2+lda-10'), True),
    ('chi2+lda-50', read_selector('reuters_t2_chi2+lda-50'), True),
    ('chi2+lda-100', read_selector('reuters_t2_chi2+lda-100'), True),
    ('chi2+lda-200', read_selector('reuters_t2_chi2+lda-200'), True),
    ('chi2+lda-300', read_selector('reuters_t2_chi2+lda-300'), True),
    ('chi2+lda-400', read_selector('reuters_t2_chi2+lda-400'), True),
    ('chi2+lda-500', read_selector('reuters_t2_chi2+lda-500'), True),
    ('chi2+lda-600', read_selector('reuters_t2_chi2+lda-600'), True),
    ('chi2+lda-700', read_selector('reuters_t2_chi2+lda-700'), True),
    ('chi2+lda-800', read_selector('reuters_t2_chi2+lda-800'), True),
    ('chi2+lda-900', read_selector('reuters_t2_chi2+lda-900'), True),
    ('chi2+lda-1000', read_selector('reuters_t2_chi2+lda-1000'), True),
    # ('lda-none', None)
]

lda_transforms_20ng_t2_fitted = [
    ('lda-10', read_selector('20ng_t2_lda-10'), True),
    ('lda-50', read_selector('20ng_t2_lda-50'), True),
    ('lda-100', read_selector('20ng_t2_lda-100'), True),
    ('lda-200', read_selector('20ng_t2_lda-200'), True),
    ('lda-300', read_selector('20ng_t2_lda-300'), True),
    ('lda-400', read_selector('20ng_t2_lda-400'), True),
    ('lda-500', read_selector('20ng_t2_lda-500'), True),
    ('lda-600', read_selector('20ng_t2_lda-600'), True),
    ('lda-700', read_selector('20ng_t2_lda-700'), True),
    ('lda-800', read_selector('20ng_t2_lda-800'), True),
    ('lda-900', read_selector('20ng_t2_lda-900'), True),
    ('lda-1000', read_selector('20ng_t2_lda-1000'), True),
    # ('lda-none', None)
]

lsi_transforms_20ng_t2_fitted = [
    ('lsi-10', read_selector('20ng_t2_lsi-10'), True),
    ('lsi-50', read_selector('20ng_t2_lsi-50'), True),
    ('lsi-100', read_selector('20ng_t2_lsi-100'), True),
    ('lsi-200', read_selector('20ng_t2_lsi-200'), True),
    ('lsi-300', read_selector('20ng_t2_lsi-300'), True),
    ('lsi-400', read_selector('20ng_t2_lsi-400'), True),
    ('lsi-500', read_selector('20ng_t2_lsi-500'), True),
    ('lsi-600', read_selector('20ng_t2_lsi-600'), True),
    ('lsi-700', read_selector('20ng_t2_lsi-700'), True),
    ('lsi-800', read_selector('20ng_t2_lsi-800'), True),
    ('lsi-900', read_selector('20ng_t2_lsi-900'), True),
    ('lsi-1000', read_selector('20ng_t2_lsi-1000'), True),
    # ('lda-none', None)
]

lda_transforms_reuters_t2_fitted = [
    ('lda-10', read_selector('reuters_t2_lda-10'), True),
    ('lda-50', read_selector('reuters_t2_lda-50'), True),
    ('lda-100', read_selector('reuters_t2_lda-100'), True),
    ('lda-200', read_selector('reuters_t2_lda-200'), True),
    ('lda-300', read_selector('reuters_t2_lda-300'), True),
    ('lda-400', read_selector('reuters_t2_lda-400'), True),
    ('lda-500', read_selector('reuters_t2_lda-500'), True),
    ('lda-600', read_selector('reuters_t2_lda-600'), True),
    ('lda-700', read_selector('reuters_t2_lda-700'), True),
    ('lda-800', read_selector('reuters_t2_lda-800'), True),
    ('lda-900', read_selector('reuters_t2_lda-900'), True),
    ('lda-1000', read_selector('reuters_t2_lda-1000'), True),
    # ('lda-none', None)
]

lsi_transforms_reuters_t2_fitted = [
    ('lsi-10', read_selector('reuters_t2_lsi-10'), True),
    ('lsi-50', read_selector('reuters_t2_lsi-50'), True),
    ('lsi-100', read_selector('reuters_t2_lsi-100'), True),
    ('lsi-200', read_selector('reuters_t2_lsi-200'), True),
    ('lsi-300', read_selector('reuters_t2_lsi-300'), True),
    ('lsi-400', read_selector('reuters_t2_lsi-400'), True),
    ('lsi-500', read_selector('reuters_t2_lsi-500'), True),
    ('lsi-600', read_selector('reuters_t2_lsi-600'), True),
    ('lsi-700', read_selector('reuters_t2_lsi-700'), True),
    ('lsi-800', read_selector('reuters_t2_lsi-800'), True),
    ('lsi-900', read_selector('reuters_t2_lsi-900'), True),
    ('lsi-1000', read_selector('reuters_t2_lsi-1000'), True),
    # ('lda-none', None)
]


class RandomProjectorWithSelector:

    def __init__(self, n_topics):
        self.model = None
        self.n_topics = n_topics
        self.selector = SelectPercentile(chi2, 10)

    def fit(self, data, target=None):
        self.selector = self.selector.fit(data, target)
        data = self.selector.fit_transform(data, target)
        self.model = SparseRandomProjection(n_components=self.n_topics)
        self.model.fit(data, target)

    def fit_transform(self, data, target=None):
        data = self.selector.fit_transform(data, target)
        self.model = SparseRandomProjection(n_components=self.n_topics)
        return self.model.fit_transform(data, target)

    def transform(self, data):
        data = self.selector.transform(data)
        return self.model.transform(data)


random_projection_with_selector_transforms_20ng_t2_fitted = [
    ('chi2+random-projection-10', read_selector('20ng_t2_chi2+random-projection-10'), True),
    ('chi2+random-projection-50', read_selector('20ng_t2_chi2+random-projection-50'), True),
    ('chi2+random-projection-100', read_selector('20ng_t2_chi2+random-projection-100'), True),
    ('chi2+random-projection-200', read_selector('20ng_t2_chi2+random-projection-200'), True),
    ('chi2+random-projection-300', read_selector('20ng_t2_chi2+random-projection-300'), True),
    ('chi2+random-projection-400', read_selector('20ng_t2_chi2+random-projection-400'), True),
    ('chi2+random-projection-500', read_selector('20ng_t2_chi2+random-projection-500'), True),
    ('chi2+random-projection-600', read_selector('20ng_t2_chi2+random-projection-600'), True),
    ('chi2+random-projection-700', read_selector('20ng_t2_chi2+random-projection-700'), True),
    ('chi2+random-projection-800', read_selector('20ng_t2_chi2+random-projection-800'), True),
    ('chi2+random-projection-900', read_selector('20ng_t2_chi2+random-projection-900'), True),
    ('chi2+random-projection-1000', read_selector('20ng_t2_chi2+random-projection-1000'), True),
    # ('random-projection-none', None)
]

random_projection_with_selector_transforms_reuters_t2_fitted = [
    ('chi2+random-projection-10', read_selector('reuters_t2_chi2+random-projection-10'), True),
    ('chi2+random-projection-50', read_selector('reuters_t2_chi2+random-projection-50'), True),
    ('chi2+random-projection-100', read_selector('reuters_t2_chi2+random-projection-100'), True),
    ('chi2+random-projection-200', read_selector('reuters_t2_chi2+random-projection-200'), True),
    ('chi2+random-projection-300', read_selector('reuters_t2_chi2+random-projection-300'), True),
    ('chi2+random-projection-400', read_selector('reuters_t2_chi2+random-projection-400'), True),
    ('chi2+random-projection-500', read_selector('reuters_t2_chi2+random-projection-500'), True),
    ('chi2+random-projection-600', read_selector('reuters_t2_chi2+random-projection-600'), True),
    ('chi2+random-projection-700', read_selector('reuters_t2_chi2+random-projection-700'), True),
    ('chi2+random-projection-800', read_selector('reuters_t2_chi2+random-projection-800'), True),
    ('chi2+random-projection-900', read_selector('reuters_t2_chi2+random-projection-900'), True),
    ('chi2+random-projection-1000', read_selector('reuters_t2_chi2+random-projection-1000'), True),
    # ('random-projection-none', None)
]

chi2_selectors_long = [
    ('chi2-select-1000', SelectKBest(chi2, 1000), False),
    ('chi2-perc-10', SelectPercentile(chi2, 10), False),
    ('chi2-perc-20', SelectPercentile(chi2, 20), False),
    ('chi2-perc-30', SelectPercentile(chi2, 30), False),
    ('chi2-perc-40', SelectPercentile(chi2, 40), False),
    ('chi2-perc-50', SelectPercentile(chi2, 50), False),
    ('chi2-perc-60', SelectPercentile(chi2, 60), False),
    ('chi2-perc-70', SelectPercentile(chi2, 70), False),
    ('chi2-perc-80', SelectPercentile(chi2, 80), False),
    ('chi2-perc-90', SelectPercentile(chi2, 90), False),
    # ('chi2-select-all', None, False)
]

variance_selectors_long = [
    ('variance-treshold-0.0001', VarianceThreshold(threshold=0.0001), False),
    ('variance-treshold-0.00001', VarianceThreshold(threshold=0.00001), False),
    ('variance-treshold-0.000001', VarianceThreshold(threshold=0.000001), False),
    ('variance-treshold-0.0000001', VarianceThreshold(threshold=0.0000001), False),
    ('variance-treshold-0.00000001', VarianceThreshold(threshold=0.00000001), False),
    # ('variance-treshold-none', None, False)
]

random_projection_transforms_reuters_long = [
    ('random-projection-1000', SparseRandomProjection(n_components=1000), False),
    ('random-projection-2000', SparseRandomProjection(n_components=2000), False),
    ('random-projection-4000', SparseRandomProjection(n_components=4000), False),
    ('random-projection-6000', SparseRandomProjection(n_components=6000), False),
    ('random-projection-8000', SparseRandomProjection(n_components=8000), False),
    ('random-projection-10000', SparseRandomProjection(n_components=10000), False),
    ('random-projection-12000', SparseRandomProjection(n_components=12000), False),
    ('random-projection-14000', SparseRandomProjection(n_components=14000), False),
    ('random-projection-16000', SparseRandomProjection(n_components=16000), False),
    ('random-projection-18000', SparseRandomProjection(n_components=18000), False),
    # ('random-projection-none', None, False)
]

selectors_to_train = [
    # (
    #     '20ng_t2',
    #     '20newsgroups',
    #     TfidfVectorizer(decode_error='ignore', stop_words='english'),
    #     lsi_transforms
    # ),
    # (
    #     'reuters_t2',
    #     'reuters',
    #     TfidfVectorizer(decode_error='ignore', stop_words='english'),
    #     lsi_transforms`
    # ),
    # (
    #     '20ng_t2',
    #     '20newsgroups',
    #     TfidfVectorizer(decode_error='ignore', stop_words='english'),
    #     lda_transforms
    # ),
    # (
    #     'reuters_t2',
    #     'reuters',
    #     TfidfVectorizer(decode_error='ignore', stop_words='english'),
    #     lda_transforms
    # ),
    # (
    #     '20ng_t2',
    #     '20newsgroups',
    #     TfidfVectorizer(decode_error='ignore', stop_words='english'),
    #     random_projection_transforms
    # ),
    # (
    #     'reuters_t2',
    #     'reuters',
    #     TfidfVectorizer(decode_error='ignore', stop_words='english'),
    #     random_projection_transforms
    # ),
    # (
    #     '20ng_t2',
    #     '20newsgroups',
    #     TfidfVectorizer(decode_error='ignore', stop_words='english'),
    #     chi2_lsi_transforms
    # ),
    # (
    #     'reuters_t2',
    #     'reuters',
    #     TfidfVectorizer(decode_error='ignore', stop_words='english'),
    #     chi2_lsi_transforms
    # ),
    # (
    #     '20ng_t2',
    #     '20newsgroups',
    #     TfidfVectorizer(decode_error='ignore', stop_words='english'),
    #     chi2_lda_transforms
    # ),
    # (
    #     'reuters_t2',
    #     'reuters',
    #     TfidfVectorizer(decode_error='ignore', stop_words='english'),
    #     chi2_lda_transforms
    # ),
    # (
    #     '20ng_t2',
    #     '20newsgroups',
    #     TfidfVectorizer(decode_error='ignore', stop_words='english'),
    #     random_projection_with_selector_transforms
    # ),
    # (
    #     'reuters_t2',
    #     'reuters',
    #     TfidfVectorizer(decode_error='ignore', stop_words='english'),
    #     random_projection_with_selector_transforms
    # ),
]


def test_selectors_top_baseline():
    classification_20newsgroups = [
        # (
        #     '20newsgroups',
        #     ('SVM-penalty-l2-dual-true-C-1', LinearSVC(penalty='l2')),
        #     ('Tfidf-removed-stopwords', TfidfVectorizer(decode_error='ignore', stop_words='english'))
        # ),
        (
            '20newsgroups',
            ('kNN-10', KNeighborsClassifier(n_neighbors=10)),
            ('Tfidf-removed-stopwords', TfidfVectorizer(decode_error='ignore', stop_words='english'))
        ),

    ]
    #
    # classification_20newsgroups_NB = [
    #     (
    #         '20newsgroups',
    #         ('MultiNB', MultinomialNB(alpha=.01)),
    #         ('Tfidf-removed-stopwords', TfidfVectorizer(decode_error='ignore', stop_words='english'))
    #     ),
    # ]

    classification_reuters = [
        # (
        #     'reuters',
        #     ('SVM-penalty-l2-dual-true-C-1', LinearSVC(penalty='l2')),
        #     ('Tfidf-removed-stopwords', TfidfVectorizer(decode_error='ignore', stop_words='english'))
        # ),

        (
            'reuters',
            ('kNN-10', KNeighborsClassifier(n_neighbors=10)),
            ('Tfidf-removed-stopwords', TfidfVectorizer(decode_error='ignore', stop_words='english'))
        ),
    ]

    # classification_reuters_NB = [
    #     (
    #         'reuters',
    #         ('MultiNB', MultinomialNB(alpha=.01)),
    #         ('Tfidf-removed-stopwords', TfidfVectorizer(decode_error='ignore', stop_words='english'))
    #     ),
    # ]

    # KROTKIE WYKRESY
    # test_selectors(classification_20newsgroups,
    #                [
    #                     # ('chi2', chi2_selectors),
    #                     # ('random-projection', random_projection_transforms_20ng_t2_fitted),
    #                     # ('lsi', lsi_transforms_20ng_t2_fitted),
    #                     # ('lda', lda_transforms_20ng_t2_fitted),
    #                     ('chi2+random-projection', random_projection_with_selector_transforms_20ng_t2_fitted),
    #                     ('chi2+lsi', chi2_lsi_transforms_20ng_t2_fitted),
    #                     ('chi2+lda', chi2_lda_transforms_20ng_t2_fitted),
    #                 ]
    # )

    # test_selectors(classification_reuters,
    #                [
    #                     ('chi2', chi2_selectors),
    #                     ('random-projection', random_projection_transforms_reuters_t2_fitted),
    #                     ('lsi', lsi_transforms_reuters_t2_fitted),
    #                     ('lda', lda_transforms_reuters_t2_fitted),
    #                     ('chi2+random-projection', random_projection_with_selector_transforms_reuters_t2_fitted),
    #                     ('chi2+lsi', chi2_lsi_transforms_reuters_t2_fitted),
    #                     ('chi2+lda', chi2_lda_transforms_reuters_t2_fitted),
    #                 ]
    # )
    #
    # test_selectors(classification_20newsgroups_NB,
    #                [
    #                     ('chi2', chi2_selectors),
    #                     # ('random-projection', random_projection_transforms),
    #                     # ('lsi', lsi_transforms_reuters_t2_fitted),
    #                     # ('lda', lda_transforms_20ng_t2_fitted),
    #                     # ('chi2+random-projection', random_projection_with_selector_transforms),
    #                     # ('chi2+lsi', chi2_lsi_transforms),
    #                     # ('chi2+lda', chi2_lda_transforms_20ng_t2_fitted),
    #                 ]
    # )
    #
    # test_selectors(classification_reuters_NB,
    #                [
    #                     ('chi2', chi2_selectors),
    #                     # ('random-projection', random_projection_transforms),
    #                     # ('lsi', lsi_transforms_reuters_t2_fitted),
    #                     # ('lda', lda_transforms_reuters_t2_fitted),
    #                     # ('chi2+random-projection', random_projection_with_selector_transforms),
    #                     # ('chi2+lsi', chi2_lsi_transforms),
    #                     # ('chi2+lda', chi2_lda_transforms_reuters_t2_fitted),
    #                 ]
    # )

    # DLUGIE WYKRESY
    test_selectors(classification_20newsgroups,
                   [
                       ('chi2', chi2_selectors),
                       # ('variance', variance_selectors_long),
                       # ('random-projection', random_projection_transforms_20ng_long),
                   ]
                   )
    #
    test_selectors(classification_reuters,
                   [
                       ('chi2', chi2_selectors),
                       # ('variance', variance_selectors_long),
                       # ('random-projection', random_projection_transforms_reuters_long),
                   ]
                   )
    #
    # # test_selectors(classification_20newsgroups_NB,
    # #                [
    # #                     ('chi2', chi2_selectors_long),
    # #                     ('variance', variance_selectors_long),
    # #                 ]
    # # )
    #
    # test_selectors(classification_reuters_NB,
    #                [
    #                     ('chi2', chi2_selectors_long),
    #                     #('variance', variance_selectors_long),
    #                 ]
    # )


# train_and_save_selectors(selectors_to_train)

test_selectors_top_baseline()
