# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import os
import sys
import json
import h5py
import pickle
import logging
import datetime
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.sparse import hstack

from sklearn.datasets import dump_svmlight_file
from sklearn.pipeline import make_pipeline, make_union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def pad_sequence(sequences, maxlen=10, padding='pre', truncating='pre', value=0):
    """
    :param sequences:
    :param maxlen:
    :param padding:
    :param truncating:
    :param value:
    :return:

     >>> sequences =[[1,3,4], [2], [1,0,3,23,65]]
     >>> pad_sequence(sequences, maxlen=3, padding='pre', truncating='pre')
     >>> [[1, 3, 4], [0, 0, 2], [3, 23, 65]]

     >>> sequences =[[1,3,4], [2], [1,0,3,23,65]]
     >>> pad_sequence(sequences, maxlen=3, padding='post', truncating='post')
     >>> [[1, 3, 4], [2, 0, 0], [1, 0, 3]]

     >>> sequences =[[1,3,4], [2], [1,0,3,23,65]]
     >>> pad_sequence(sequences, maxlen=10, padding='post', truncating='post', value='copy')
     >>> [[1, 3, 4, 1, 3, 4, 1, 3, 4, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 0, 3, 23, 65, 1, 0, 3, 23, 65]]
    """

    padded_sequences = []

    for seq in sequences:

        length = len(seq)

        if truncating == 'pre':
            seq = seq[-maxlen:]
        elif truncating == 'post':
            seq = seq[:maxlen]

        if length < maxlen:

            diff = np.abs(length - maxlen)

            if padding == 'pre':

                if type(value) == int:
                    seq = [value] * diff + seq
                elif value == 'copy':

                    seq = [0] + seq

                    if length == 0:
                        seq = [0] * maxlen
                    else:
                        n_copy = int(np.ceil((maxlen / (len(seq)))))
                        seq = (seq * n_copy)[-maxlen:]

            elif padding == 'post':
                if type(value) == int:
                    seq = [value] * diff + seq
                elif value == 'copy':
                    seq = [0] + seq

                    if length == 0:
                        seq = [0] * maxlen
                    else:
                        n_copy = int(np.ceil((maxlen / (len(seq)))))
                        seq = (seq * n_copy)[:maxlen]

        padded_sequences.append(seq)
    return padded_sequences


def try_access(df, cols):
    try:
        return df[cols]
    except KeyError:
        sys.exit("name: '{}' can't be found".format(cols))


class TextVectorizer(object):
    """
    Turn a list of strings into a list of integers
    train = ["hello brosss", "XDLOL"]
    test = ["unseen TOKEN", "see you"]

    vec = StringToSequence()
    vec.fit(train)

    train_seq = vec.transform(train)
    test_seq = vec.transform(test)

    print(train_seq)
    print(test_seq)
    print(vec.inverse_tranform(train_seq))
    print(vec.inverse_tranform(test_seq))

    [[67, 62, 23, 23, 34, 57, 16, 34, 41, 41, 41], []]
    [[60, 13, 41, 62, 62, 13], [41, 62, 62, 20, 34, 60]]
    ['hellobrosss', '']
    ['unseen', 'seeyou']
    """

    def __init__(self, alphabet="""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"""):

        self.alphabet = alphabet

        if self.alphabet:
            self.token_indice = {k: v + 1 for v, k in enumerate(set(self.alphabet))}
            self.indice_token = {self.token_indice[k]: k for k in self.token_indice}
        else:
            self.token_indice = None
            self.indice_token = None

    def _build_dictionary(self, sentences):

        dictionary = set()

        for sent in sentences:
            dictionary.update(set(sent))
        return dictionary

    def transform(self, sentences, token_indice=None):

        token_indice = token_indice if token_indice else self.token_indice

        new_sentences = []
        for sent in sentences:
            new_sentences.append([token_indice[char] for char in sent if char in token_indice])
        return new_sentences

    def inverse_tranform(self, sequences, indice_token=None):

        indice_token = indice_token if indice_token else self.indice_token
        new_sequences = []
        for seq in sequences:
            new_sequences.append("".join([indice_token[val] for val in seq if val in indice_token]))
        return new_sequences

    def fit(self, sentences, y=None):

        dictionary = set(self.alphabet) if self.alphabet else self.build_dictionary(sentences)

        self.token_indice = {k: v + 1 for v, k in enumerate(dictionary)}
        self.indice_token = {self.token_indice[k]: k for k in self.token_indice}

        return self

    def fit_transform(self, sentences):
        self.fit(sentences)
        return self.transform(sentences, self.token_indice)


class ContinuousVectorizers(object):
    def __init__(self, columns=[], clip_values=True, scale=True):

        self.columns = columns
        self.clip_values = clip_values
        self.scale = scale

        self.means = np.zeros(len(self.columns))
        self.stds = np.ones(len(self.columns))

    def fit(self, df, y=None):

        if self.clip_values:
            df = df[self.columns].clip(lower=-1e6, upper=1e6)

        for i, col in enumerate(self.columns):

            dfcol = df[col]

            if self.scale:
                mean, std = dfcol.mean(), dfcol.std()
                if std.sum() == 0:
                    if mean.sum() == 0:
                        print("warning: column: '{}', all the values are zeros".format(col))
                    else:
                        print("warning: column: '{}', all the values are identical".format(col))

                self.means[i] = mean
                self.stds[i] = std

    def transform(self, df):

        if self.clip_values:
            df = df[self.columns].clip(lower=-1e6, upper=1e6)

        x = np.zeros((len(df), len(self.columns)))

        for i, col in enumerate(self.columns):
            xcol = df[col].values
            xcol -= self.means[i]

            if self.stds[i] > 0:
                xcol /= self.stds[i]
            elif self.means[i] != 0:
                xcol /= self.means[i]

            x[:, i] = xcol

        return x

    def fit_transform(self, df, y=None):
        self.fit(df)
        return self.transform(df)


class EmbeddingVectorizers(object):
    def __init__(self, columns=[], rejected_index=0, same_space=True):
        self.columns = columns
        self.map_values_indexes = {}
        self.rejected_index = rejected_index
        self.same_space = same_space

    def fit(self, df, y=None, **fit_params):
        self.map_values_indexes = {}
        idx = 0

        for col in self.columns:

            unique_values = df[col].unique()
            map_val_idx = {}
            for val in unique_values:
                val = str(val)
                map_val_idx[val] = idx + 1
                idx += 1

            if not self.same_space:
                idx = 0
            self.map_values_indexes[col] = map_val_idx

    def transform(self, df):
        x = np.zeros((len(df), len(self.columns)), dtype=np.int)
        for col_idx, col in enumerate(self.columns):
            for row_idx, row in enumerate(df[col].apply(str).values):
                if row in self.map_values_indexes[col]:
                    x[row_idx, col_idx] = self.map_values_indexes[col][row]
                else:
                    x[row_idx, col_idx] = self.rejected_index
        return x

    def fit_transform(self, df, y=None):
        self.fit(df)
        return self.transform(df)


class DummyVectorizer(BaseEstimator):
    def __init__(self, max_features=None, min_count=1, max_count=1.,
                 return_sparse=True, reject_col_name='OTHER', dtype=np.int8, prefix=None):
        """Make one hot encoding of features

        Parameters
        ----------
        max_features : int
            Number of maximum features. Will take top features
             and other will put as label 'OTHER'
        min_count : int
            Lower bound to filter dataset
        max_count : int
            Top bound to filter dataset
        return_sparse : boolean
            Return dense or sparse data, default True
        dtype : numpy.type
            In which type return values

        Returns
        -------
        sparse or dense
            numpy.ndarray or pandas.Series of required columns

        """
        self.max_features = max_features
        self.min_count = min_count
        self.max_count = max_count
        self.return_sparse = return_sparse
        self.dtype = dtype
        self.prefix = prefix
        self.feature_names = None
        self.vocabulary_ = None
        self.reject_col_name = reject_col_name
        self.vectorizer = CountVectorizer(encoding='utf-8', strip_accents=None,
                                          lowercase=False,
                                          analyzer='word',
                                          token_pattern="^.*$",
                                          preprocessor=self.prepro,
                                          max_features=None,
                                          max_df=self.max_count,
                                          min_df=self.min_count,
                                          dtype=dtype)

    def _transform_to_top(self, X):
        x = np.ravel(X)
        if self.vocabulary_ is None:
            # Get label - counts
            label, counts = np.unique(x, return_counts=True)
            # Create label:count dictionary
            d = dict(zip(label, counts))
            # Extract top frequent labels
            self.vocabulary_ = sorted(d, key=d.get, reverse=True)[:self.max_features]
        # Replace labels that are not in feature_name by token 'OTHER'
        for i, label in enumerate(x):
            x[i] = label if (label in self.vocabulary_) else self.reject_col_name

        return x

    def fit(self, X, y=None, **fit_params):
        self.feature_names = None
        if self.max_features is None:
            self.vectorizer.fit(np.ravel(X))
        else:
            self.vectorizer.fit(self._transform_to_top(X))
        self.vocabulary_ = self.vectorizer.get_feature_names()
        self.feature_names = ['{}_{}'.format(self.prefix, fn) for fn in self.vocabulary_]
        return self

    def transform(self, X):
        x = self.vectorizer.transform(self._transform_to_top(X))
        if not self.return_sparse:
            x = x.toarray()
        return x

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        x = self.transform(X)
        if not self.return_sparse:
            x = x.toarray()
        return x

    def prepro(self, x):
        return str(x)


def ddify_csv(input_csv='news20.csv',
              output_folder='test_ddify_csv',
              output_csv="news20-vectorized.csv",
              maxlen=1024, lowercase=False, same_space=False, padding='pre', padding_value=0, scale_continuous=False,
              txt_fields=['content_body'],
              cnt_fields=['user_customerSpecific_textLikeCount', 'user_customerSpecific_textLikeCount'],
              cat_fields=['user_customerSpecific_seek', 'user_customerSpecific_gender'],
              label_field='content_customerSpecific_moderationId',
              alphabet="""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""",
              word_vectors_path="",
              txt_mapping_path="",
              cnt_mapping_path="",
              cat_mapping_path="",
              lab_mapping_path=""):

    dataset = []
    indexes = {'label': '', 'txt': '', 'cnt': '', 'cat': ''}
    fmts = []
    curr_idx = 0

    os.makedirs(output_folder, exist_ok=True)

    print("-> reading {}".format(os.path.join(output_folder, input_csv)))
    df = pd.read_csv(input_csv)

    labels = try_access(df, label_field).values
    print("-> processing labels")

    ratio = try_access(df, label_field).apply(str).value_counts().apply(lambda r: r / len(df)).to_dict()
    fp = os.path.join(output_folder, "lab_stats.json")
    print(" - calculate ratio and dump: {}".format(fp))
    with open(fp, "w") as f:
        json.dump(ratio, f, indent=2)

    le = LabelEncoder()

    if lab_mapping_path:
        print("   - loading from: {}".format(lab_mapping_path))
        mapping = json.loads(open(lab_mapping_path).read())
        classes = [mapping[str(i)] for i in range(len(mapping))]
        le.classes_ = np.array(classes)
    else:
        le.fit(labels)
        mapping = {map: real for map, real in enumerate(le.classes_.tolist())}
        with open(os.path.join(output_folder, "lab_mapping.json"), "w") as f:
            json.dump(mapping, f, indent=2)

    labels = le.transform(labels)

    dataset.append(labels.reshape(-1, 1))
    indexes['label'] = 0
    fmts.append("%.d")
    curr_idx += 1

    if txt_fields:
        print("-> processing text columns: {}".format(txt_fields))

        if not isinstance(txt_fields, list):
            print("'txt_fields' should be a list")
            sys.exit(0)

        cols = txt_fields
        print("  - columns: {}".format(cols))
        df_txt = try_access(df, cols)

        if len(cols) == 1:
            sentences = df_txt[cols[0]].apply(str).tolist()
        elif len(cols) == 2:
            sentences = (df_txt[cols[0]] + " " + df_txt[cols[1]]).apply(str).tolist()
        else:
            print("'txt_fields' with more that 2 elements is not supported yet")
            sys.exit(0)

        if lowercase:
            print("  - lowering sentences")
            sentences = [sent.lower() for sent in sentences]

        print("  - vectorizing text")
        if txt_mapping_path:
            print("   - loading from: {}".format(txt_mapping_path))
            mapping = json.loads(open(txt_mapping_path).read())
            vec = TextVectorizer()
            vec.token_indice = mapping

        else:
            if alphabet:
                alphabet = alphabet
                vec = TextVectorizer(alphabet=alphabet)
            else:
                vec = TextVectorizer(alphabet=None)
                vec.fit(sentences)

            mapping = vec.token_indice

            with open(os.path.join(output_folder, "txt_mapping.json"), "w") as f:
                json.dump(mapping, f, indent=2)

        x = vec.transform(sentences)

        print("  - padding")
        x = np.array(pad_sequence(x, maxlen=maxlen, padding=padding, value=padding_value))

        dataset.append(x)
        indexes['txt'] = "{}-{}".format(curr_idx, curr_idx + maxlen - 1)
        fmt = ["%.1f"] * maxlen
        fmts.extend(fmt)
        curr_idx += maxlen

        if os.path.exists(word_vectors_path):
            print("  - adding word vectors: {}".format(word_vectors_path))
            import gensim
            model = gensim.models.KeyedVectors.load_word2vec_format(word_vectors_path, binary=True)

            x = np.zeros((len(sentences), model.vector_size))

            for i, sent in enumerate(sentences):
                for word in sent.split(" "):
                    if word in model:
                        x[i, :] += model[word]

            dataset.append(x)

            indexes['word_embedding'] = "{}-{}".format(curr_idx, curr_idx + model.vector_size - 1)
            fmt = ["%.4f"] * x.shape[1]
            fmts.extend(fmt)
            curr_idx += x.shape[1]

    if cnt_fields:
        print("-> processing continuous columns: {}".format(cnt_fields))

        cols = cnt_fields
        array = try_access(df, cols)

        vec = ContinuousVectorizers(columns=cols, scale=scale_continuous)

        if cnt_mapping_path:
            print("   - loading from: {}".format(cnt_mapping_path))
            mapping = json.loads(open(cnt_mapping_path).read())

            if scale_continuous:
                vec.means = [mapping[col]['mean'] for col in cols]
                vec.stds = [mapping[col]['std'] for col in cols]

        else:
            vec.fit(array)
            mapping = {}
            for i, col in enumerate(cols):

                if scale_continuous:
                    mapping[col] = {'index': curr_idx, 'mean': vec.means[i], 'std': vec.stds[i]}
                else:
                    mapping[col] = {'index': curr_idx, 'mean': 0, 'std': 1}
                curr_idx += 1

            with open(os.path.join(output_folder, "cnt_mapping.json"), "w") as f:
                json.dump(mapping, f, indent=2)

        x = vec.transform(array)
        dataset.append(x)
        indexes['cnt'] = "{}-{}".format(curr_idx, curr_idx + x.shape[1] - 1)
        fmts.extend(["%.6f"] * x.shape[1])
        curr_idx += x.shape[1]

    if cat_fields:
        print("-> processing categorical columns: {}".format(cat_fields))
        cols = cat_fields
        array = try_access(df, cat_fields)

        vec = EmbeddingVectorizers(columns=cols, same_space=same_space)

        if cat_mapping_path:
            print("   - loading from: {}".format(cat_mapping_path))
            mapping = json.loads(open(cat_mapping_path).read())
            vec.map_values_indexes = {col: mapping[col]['values'] for col in cols}

        else:
            vec.fit(array)
            mapping = {}
            for i, col in enumerate(cols):
                mapping[col] = {'index': curr_idx, 'values': vec.map_values_indexes[col]}
                curr_idx += 1

            with open(os.path.join(output_folder, "cat_mapping.json"), "w") as f:
                json.dump(mapping, f, indent=2)

        x = vec.transform(array)
        dataset.append(x)
        indexes['cat'] = "{}-{}".format(curr_idx, curr_idx + x.shape[1] - 1)
        embedding_max_idx = int(
            np.max([np.max([k for k in vec.map_values_indexes[col].values()]) for col in vec.map_values_indexes]))
        print(embedding_max_idx)
        indexes['embedding_max_index'] = embedding_max_idx

        fmts.extend(["%.d"] * x.shape[1])
        curr_idx += x.shape[1]

    print("-> saving datasets to: {}".format(os.path.join(output_folder, output_csv)))
    dataset = np.concatenate(dataset, 1)
    np.savetxt(os.path.join(output_folder, output_csv), dataset, delimiter=',', fmt=fmts,
               header=",".join([str(x) for x in range(len(fmts))]), comments='')

    print("-> saving indexes")
    with open(os.path.join(output_folder, "indexes.json"), "w") as f:
        json.dump(indexes, f, indent=2)


def ddify_csv_args():
    parser = argparse.ArgumentParser("Parameters to vectorize csv dataset")

    parser.add_argument("--input_csv", type=str, default="/mnt/terabox/research/nlp-benchmarks/datasets/affinitas/affinitas-en-0/csv/train.csv")
    parser.add_argument("--output_folder", type=str, default="test_ddify_csv")
    parser.add_argument("--output_csv", type=str, default="train-vectorized.csv")

    parser.add_argument("--maxlen", type=int, default=1024)
    parser.add_argument("--lowercase", type=bool, default=False)
    parser.add_argument("--same_space", type=bool, default=True)
    parser.add_argument("--padding", type=str, default="pre")
    parser.add_argument("--padding_value", type=int, default=0)
    parser.add_argument("--scale_continuous", type=bool, default=True)

    parser.add_argument("--txt_fields", nargs='+', type=str, default=["content_title", "content_body"])
    parser.add_argument("--cnt_fields", nargs='+', type=str, default=['user_customerSpecific_textLikeCount', 'user_customerSpecific_textLikeCount'])
    parser.add_argument("--cat_fields", nargs='+', type=str, default=['user_customerSpecific_seek', 'user_customerSpecific_gender'])
    parser.add_argument("--label_field", type=str, default="content_customerSpecific_moderationId")
    parser.add_argument("--alphabet", type=str, default="""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")

    parser.add_argument("--word_vectors_path", type=str, default="/home/ardalan.mehrani/projects/nlp-benchmarks/GoogleNews-vectors-negative300.bin")
    parser.add_argument("--txt_mapping_path", type=str, default="")
    parser.add_argument("--cnt_mapping_path", type=str, default="")
    parser.add_argument("--cat_mapping_path", type=str, default="")
    parser.add_argument("--lab_mapping_path", type=str, default="")
    args = parser.parse_args()
    return args


def vectorize_csv(opt=None):

    opt = opt if opt else ddify_csv_args()

    ddify_csv(**vars(opt))


if __name__ == "__main__":
    pass


