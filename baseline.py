from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer, HashingVectorizer
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, f_classif
from sklearn import random_projection
from sklearn.decomposition import PCA, NMF
from sklearn.lda import LDA
# from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import mutual_info_score
from sklearn.utils import atleast2d_or_csr
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import array2d
from sklearn.preprocessing import LabelBinarizer
from gensim import models, matutils
from datetime import datetime
import numpy as np
from scipy.sparse import issparse
from sklearn.feature_selection import VarianceThreshold

# reuters
from nltk.corpus import reuters
from collections import namedtuple
import re

from pprint import pprint

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def get_data(subset, corpus='20newsgroups'):
    if 'reuters' == corpus:
        categories = [category for category in reuters.categories() if len(reuters.fileids(category)) > 100]
        # categories = ['acq', 'bop', 'coffee', 'corn', 'crude', 'dlr']
        categories = []

        if 'train' == subset:
            pattern = 'training'
        else:
            pattern = 'test'

        data = [(fileid, reuters.categories(fileid)) for fileid in reuters.fileids()]
        subset_getter = re.compile(pattern + '.*')
        data = [(reuters.raw(document[0]), document[1][0]) for document in data
                if subset_getter.match(document[0]) and len(document[1]) == 1]
        data = [document for document in data if document[1] in categories]

        data = np.array(data)
        Result = namedtuple('result', 'data target')
        return Result(data[:, 0], data[:, 1])

    else:
        categories = ['alt.atheism', 'comp.graphics']
        # categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
        # categories = None  # means all
        remove = ['headers', 'footers', 'quotes']
        # remove = []

        # return fetch_20newsgroups(subset=subset, categories=categories, shuffle=True, random_state=42, remove=remove)
        return fetch_20newsgroups(subset=subset, categories=categories, remove=remove)


def time_it(comment, func, *args):
    start_time = datetime.now()
    result = func(*args)
    print (datetime.now() - start_time), ' - ', comment
    return result


def time_it_loud(comment, func, *args):
    start_time = datetime.now()
    result = func(*args)
    print (datetime.now() - start_time), ' - ', comment
    return result


def information_gain(x, y):
    def entropy(labels):
        counts = np.bincount(labels)
        probs = counts[np.nonzero(counts)] / float(len(labels))
        return - np.sum(probs * np.log(probs))

    entropy_before = entropy(y)

    def _information_gain(x, y):
        # feature_number = x.shape[1]
        feature_number = len(x)
        # x_set_indices = np.nonzero(x)[1]
        x_set_indices = np.nonzero(x)[0]
        x_not_set_indices = [i for i in range(0, feature_number) if i not in x_set_indices]

        entropy_x_set = entropy(y[x_set_indices])
        entropy_x_not_set = entropy(y[x_not_set_indices])

        return entropy_before - (((len(x_set_indices) / float(feature_number)) * entropy_x_set)
                                 + ((len(x_not_set_indices) / float(feature_number)) * entropy_x_not_set))

    feature_number = x.shape[0]
    feature_range = range(0, feature_number)
    x = x.toarray()
    information_gain_scores = []
    for feature in x.T:
        # feature_number = feature.shape[1]
        # feature_number = len(feature)
        # x_set_indices = np.nonzero(feature)[1]
        x_set_indices = np.nonzero(feature)[0]
        x_not_set_indices = [i for i in feature_range if i not in x_set_indices]

        labels = y[x_set_indices]
        counts = np.bincount(labels)
        probs = counts[np.nonzero(counts)] / float(len(labels))
        entropy_x_set = - np.sum(probs * np.log(probs))

        labels = y[x_not_set_indices]
        counts = np.bincount(labels)
        probs = counts[np.nonzero(counts)] / float(len(labels))
        entropy_x_not_set = - np.sum(probs * np.log(probs))

        len_x_set_indices = len(x_set_indices)
        len_x_not_set_indices = feature_number - len_x_set_indices

        information_gain_scores.append(entropy_before
                                       - (((len_x_set_indices / float(feature_number)) * entropy_x_set)
                                          + ((len_x_not_set_indices / float(feature_number)) * entropy_x_not_set)))

        # information_gain_scores.append(_information_gain(feature, y))

    # print information_gain_scores[-1000:]
    return information_gain_scores, []


def information_gain_matrix(X, y):
    def entropy(labels, s):
        probs = labels / float(s)
        return - np.sum(probs * np.log(probs))

    Y = LabelBinarizer().fit_transform(y)
    if Y.shape[1] == 1:
        Y = np.append(1 - Y, Y, axis=1)

    X = X.toarray()
    # print 'Observed:'
    ones = np.ones((X.shape[0], X.shape[1] + 1))
    ones[:, :-1] = X
    X = ones

    X = atleast2d_or_csr(X)
    observed = safe_sparse_dot(Y.T, X)  # n_classes * n_features
    # pprint(observed)

    observed = observed.T
    counts_classes = observed[-1]
    entropy_before = entropy(counts_classes, sum(counts_classes))
    print 'entropy before', entropy_before
    information_gain_scores = []
    for counts in observed:
        mask = counts != 0
        classes_with_feature = counts[mask]
        sum_classes_with_feature = sum(classes_with_feature)
        entropy_classes_with_feature = entropy(classes_with_feature, sum_classes_with_feature)

        classes_without_feature = counts_classes[-mask]
        sum_classes_without_feature = sum(classes_without_feature)
        entropy_classes_without_feature = entropy(classes_without_feature, sum_classes_without_feature)

        print entropy_classes_with_feature
        print entropy_classes_without_feature

        s = sum_classes_with_feature + sum_classes_without_feature
        # print 'sum_classes_with_feature', sum_classes_with_feature
        # print 'sum_classes_without_feature', sum_classes_without_feature
        # print 's', s
        information_gain_scores.append(
            entropy_before - ((sum_classes_with_feature / float(s)) * entropy_classes_with_feature
                              + (sum_classes_without_feature / float(s)) * entropy_classes_without_feature))
    print information_gain_scores
    return information_gain_scores[:-1], []


def information_gain_profile(x, y):
    def _information_gain_SO_profile(x, y):
        def _entropy(values):
            counts = np.bincount(values)
            probs = counts[np.nonzero(counts)] / float(len(values))
            return - np.sum(probs * np.log(probs))

        feature_size = x.shape[0]
        feature_range = range(0, feature_size)
        entropy_before = _entropy(y)
        information_gain_scores = []

        import sys
        import cProfile
        cProfile.runctx(
            """for feature in x.T:
            feature_set_indices = np.nonzero(feature)[1]
            feature_not_set_indices = [i for i in feature_range if i not in feature_set_indices]

            values = y[feature_set_indices]
            counts = np.bincount(values)
            probs = counts[np.nonzero(counts)] / float(len(values))
            entropy_x_set = - np.sum(probs * np.log(probs))

            values = y[feature_not_set_indices]
            counts = np.bincount(values)
            probs = counts[np.nonzero(counts)] / float(len(values))
            entropy_x_not_set = - np.sum(probs * np.log(probs))

            result = entropy_before - (((len(feature_set_indices) / float(feature_size)) * entropy_x_set)
                                     + ((len(feature_not_set_indices) / float(feature_size)) * entropy_x_not_set))
            information_gain_scores.append(result)""",
            globals(), locals())

        sys.exit()
        return information_gain_scores, []

    return _information_gain_SO_profile(x, y)


classifiers = [
    # LinearSVC(penalty="l1", dual=False, tol=1e-3),
    LinearSVC(penalty="l2", dual=False, tol=1e-3),

    # MultinomialNB(alpha=.01),
    # BernoulliNB(alpha=.01),

    # KNeighborsClassifier(n_neighbors=5),
    # KNeighborsClassifier(n_neighbors=10),
    # KNeighborsClassifier(n_neighbors=45),
    # KNeighborsClassifier(n_neighbors=100),

    # NearestCentroid  # Rocchio
]
# classifiers = [KNeighborsClassifier(n_neighbors=5)]
extractors = [
    # CountVectorizer(stop_words='english'),
    # CountVectorizer()

    # Tfidf takes norm=l2 -- Euclidean
    # smooth_idf=True, sublinear_tf=False

    # TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english', ngram_range=(1, 1), decode_error='ignore'),
    TfidfVectorizer(),
    # HashingVectorizer(stop_words='english', non_negative=True, n_features=2 ** 16)
]
# reducers = [HashingVectorizer(non_negative=True)]
selectors = [
    # PCA()

    # SelectPercentile(chi2, 0.1),
    # SelectPercentile(chi2, 0.5),
    # SelectPercentile(information_gain_SO_profile, 1),
    # SelectPercentile(chi2, 5),
    # SelectPercentile(chi2, 1),
    SelectPercentile(f_classif, 10),
    # SelectPercentile(chi2, 10),
    # SelectPercentile(chi2, 20),
    # SelectPercentile(chi2, 30),
    # SelectPercentile(chi2, 40),
    # SelectPercentile(chi2, 50),
    # SelectPercentile(chi2, 60),
    # SelectPercentile(chi2, 70),
    # SelectPercentile(chi2, 80),
    # SelectPercentile(chi2, 90),

    # SelectKBest(chi2, k=10),
    # SelectKBest(chi2, k=100),
    # SelectKBest(chi2, k=1000),
    # SelectKBest(chi2, k=10000),
    # SelectKBest(chi2, k=100000),
    # SelectKBest(chi2, k='all'),

    # VarianceThreshold(threshold=0.00001)
]


# TODO: ekstraktory cech: LDA, LSA, PCA
# TODO: corpus reuters?


def run():
    for classifier in classifiers:
        for extractor in extractors:
            for selector in selectors:
                print 'Classifier: ', classifier
                print 'Extractor: ', extractor
                print 'Selector: ', selector

                # tfidf = TfidfTransformer(sublinear_tf=True)
                # pca = LDA(n_components=1000)

                # TRAIN

                print 'TRAINING'

                data_train = time_it('getting data', get_data, 'train')
                data_train_transformed = time_it('extractor transforming and fitting', extractor.fit_transform,
                                                 data_train.data)

                # for data in data_train_transformed:
                #    print data[0][0]

                print 'Extracted data shape ', data_train_transformed.shape
                data_train_transformed = time_it_loud('selector transforming and fitting', selector.fit_transform,
                                                      data_train_transformed, data_train.target)
                print data_train_transformed

                print 'Selected data shape ', data_train_transformed.shape
                # print 'Selected features: '
                # print np.asarray(extractor.get_feature_names())[selector.get_support()]

                """
                print 'gensim1'
                gensim_data_train_transformed = matutils.Sparse2Corpus(data_train_transformed, documents_columns=False)
                print 'gensim2'
                lsi = models.LsiModel(gensim_data_train_transformed, num_topics=162) # initialize an LSI transformation
                print 'gensim3'
                data_train_transformed = matutils.corpus2csc(lsi[gensim_data_train_transformed]).T # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
                print 'gensim4'
                """

                """
                rp = random_projection.SparseRandomProjection(n_components=10000)
                data_train_transformed = rp.fit_transform(data_train_transformed)
                print data_train_transformed.shape
                """

                # data_train_transformed = tfidf.fit_transform(data_train_transformed)
                # data_train_transformed = pca.fit_transform(data_train_transformed.toarray(), data_train.target)
                # print 'PCA data shape ', data_train_transformed.shape

                time_it_loud('classifier fitting', classifier.fit, data_train_transformed, data_train.target)

                # TEST

                print 'TESTING'

                data_test = time_it('getting data', get_data, 'test')
                data_test_transformed = time_it('extractor transforming', extractor.transform, data_test.data)
                print 'Extracted data shape ', data_test_transformed.shape
                data_test_transformed = time_it_loud('selector transforming', selector.transform, data_test_transformed)
                print 'Selected data shape ', data_test_transformed.shape

                """
                print 'gensim5'
                gensim_data_test_transformed = matutils.Sparse2Corpus(data_test_transformed, documents_columns=False)
                print 'gensim6'
                data_test_transformed = matutils.corpus2csc(lsi[gensim_data_test_transformed]).T # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
                print 'gensim7'
                """

                """
                data_test_transformed = rp.transform(data_test_transformed)
                """

                # data_test_transformed = tfidf.transform(data_test_transformed)
                # data_test_transformed = pca.transform(data_test_transformed)
                # print 'PCA data shape ', data_test_transformed.shape

                prediction = time_it_loud('classifier prediction', classifier.predict, data_test_transformed)

                # RESULT

                print 'RESULT'

                print 'F1:', metrics.f1_score(data_test.target, prediction, average='macro')
                print 'Precision', metrics.precision_score(data_test.target, prediction, average='macro')
                print 'Recall', metrics.recall_score(data_test.target, prediction, average='macro')
                print 'Accuracy', metrics.accuracy_score(data_test.target, prediction), '\n'

                # CONFUSION MATRIX

                from sklearn.metrics import confusion_matrix

                import matplotlib.pyplot as plt

                # Compute confusion matrix
                cm = confusion_matrix(data_test.target, prediction)

                print(cm)

                # Show confusion matrix in a separate window
                plt.matshow(cm)
                plt.title('Confusion matrix')
                plt.colorbar()
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                plt.show()



# import cProfile
# cProfile.run('run()')


run()
