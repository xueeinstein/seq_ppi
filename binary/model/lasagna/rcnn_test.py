from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
if '../../../embeddings' not in sys.path:
    sys.path.append('../../../embeddings')

import h5py
import hashlib
import numpy as np
from tqdm import tqdm

from seq2tensor import s2t
import keras

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM, Bidirectional, BatchNormalization, merge, add
from keras.layers.core import Flatten, Reshape
from keras.layers.merge import Concatenate, concatenate, subtract, multiply
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D

import os
# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from keras.backend import set_session

def get_emb_dim(h5_file):
    f = h5py.File(h5_file, 'r')
    k = list(f.keys())[0]
    emb_dim = np.array(f[k]).shape[-1]
    f.close()
    return emb_dim

def load_emb(h5, seq, length):
    md5 = hashlib.md5(seq.encode()).hexdigest()
    emb = np.array(h5[md5])
    if len(emb) > length:
        return emb[:length, :]
    else:
        emb_ = np.pad(emb, ((0, length-len(emb)), (0, 0)))
        return emb_

def get_session(gpu_fraction=0.5):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# class DataGenerator(keras.utils.Sequence):
class DataGenerator(keras.utils.all_utils.Sequence):
    def __init__(self, h5_file, seq_array, seq1_ids, seq2_ids, seq_len, batch_size, shuffle=True):
        assert len(seq1_ids) == len(seq2_ids)
        self.h5 = h5py.File(h5_file, 'r')
        self.seqs_1 = [seq_array[i] for i in seq1_ids]
        self.seqs_2 = [seq_array[i] for i in seq2_ids]
        self.seq_len = seq_len

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.seqs_1) / self.batch_size))

    def __getitem__(self, idx):
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]

        X1, X2 = [], []
        for i in indexes:
            X1.append(load_emb(self.h5, self.seqs_1[i], self.seq_len))
            X2.append(load_emb(self.h5, self.seqs_2[i], self.seq_len))

        X1 = np.array(X1)
        X2 = np.array(X2)
        return [X1, X2]

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.seqs_1))
        if self.shuffle:
            np.random.shuffle(self.indexes)

# KTF.set_session(get_session())
set_session(get_session())

from keras.layers import Input, CuDNNGRU
from numpy import linalg as LA
import scipy

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
# Note: if you use another PPI dataset, this needs to be changed to a corresponding dictionary file.
# id2seq_file = '../../../yeast/preprocessed/protein.dictionary.tsv'

# seq_size = 2000
seq_size = 1000
emb_files = ['../../../embeddings/default_onehot.txt', '../../../embeddings/string_vec5.txt', '../../../embeddings/CTCoding_onehot.txt', '../../../embeddings/vec5_CTC.txt']
hidden_dim = 25

# ds_file, label_index, rst_file, use_emb, hidden_dim
ds_file = '../../../yeast/preprocessed/Supp-AB.tsv'
label_index = 2
rst_file = 'results/15k_onehot_cnn.txt'
sid1_index = 0
sid2_index = 1
id2seq_file = '../../../data/human_dict.tsv'

if len(sys.argv) > 1:
    ds_file, label_index, rst_file, emb_h5, hidden_dim, id2seq_file, out_file = sys.argv[1:]
    label_index = int(label_index)
    hidden_dim = int(hidden_dim)

id2index = {}
seqs = []
index = 0
for line in open(id2seq_file):
    line = line.strip().split('\t')
    id2index[line[0]] = index
    seqs.append(line[1])
    index += 1
seq_array = []
id2_aid = {}
sid = 0

max_data = -1
limit_data = max_data > 0
raw_data = []
skip_head = True
x = None
count = 0
for line in tqdm(open(ds_file)):
    if skip_head:
        skip_head = False
        continue
    line = line.rstrip('\n').rstrip('\r').split('\t')
    if id2index.get(line[sid1_index]) is None or id2index.get(line[sid2_index]) is None:
        continue
    if id2_aid.get(line[sid1_index]) is None:
        id2_aid[line[sid1_index]] = sid
        sid += 1
        seq_array.append(seqs[id2index[line[sid1_index]]])
    line[sid1_index] = id2_aid[line[sid1_index]]
    if id2_aid.get(line[sid2_index]) is None:
        id2_aid[line[sid2_index]] = sid
        sid += 1
        seq_array.append(seqs[id2index[line[sid2_index]]])
    line[sid2_index] = id2_aid[line[sid2_index]]
    raw_data.append(line)
    if limit_data:
        count += 1
        if count >= max_data:
            break
print (len(raw_data))

len_m_seq = np.array([len(line.split()) for line in seq_array])
avg_m_seq = int(np.average(len_m_seq)) + 1
max_m_seq = max(len_m_seq)
print (avg_m_seq, max_m_seq)

dim = get_emb_dim(emb_h5)

seq_index1 = np.array([line[sid1_index] for line in tqdm(raw_data)])
seq_index2 = np.array([line[sid2_index] for line in tqdm(raw_data)])

print(seq_index1[:10])

class_map = {'0':1,'1':0}
print(class_map)
class_labels = np.zeros((len(raw_data), 2))
for i in range(len(raw_data)):
    class_labels[i][class_map[raw_data[i][label_index][0]]] = 1.

def build_model():
    seq_input1 = Input(shape=(seq_size, dim), name='seq1')
    seq_input2 = Input(shape=(seq_size, dim), name='seq2')
    l1=Conv1D(hidden_dim, 3)
    r1=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l2=Conv1D(hidden_dim, 3)
    r2=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l3=Conv1D(hidden_dim, 3)
    r3=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l4=Conv1D(hidden_dim, 3)
    r4=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l5=Conv1D(hidden_dim, 3)
    r5=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l6=Conv1D(hidden_dim, 3)
    s1=MaxPooling1D(3)(l1(seq_input1))
    s1=concatenate([r1(s1), s1])
    s1=MaxPooling1D(3)(l2(s1))
    s1=concatenate([r2(s1), s1])
    s1=MaxPooling1D(3)(l3(s1))
    s1=concatenate([r3(s1), s1])
    s1=MaxPooling1D(3)(l4(s1))
    s1=concatenate([r4(s1), s1])
    s1=MaxPooling1D(3)(l5(s1))
    s1=concatenate([r5(s1), s1])
    s1=l6(s1)
    s1=GlobalAveragePooling1D()(s1)
    s2=MaxPooling1D(3)(l1(seq_input2))
    s2=concatenate([r1(s2), s2])
    s2=MaxPooling1D(3)(l2(s2))
    s2=concatenate([r2(s2), s2])
    s2=MaxPooling1D(3)(l3(s2))
    s2=concatenate([r3(s2), s2])
    s2=MaxPooling1D(3)(l4(s2))
    s2=concatenate([r4(s2), s2])
    s2=MaxPooling1D(3)(l5(s2))
    s2=concatenate([r5(s2), s2])
    s2=l6(s2)
    s2=GlobalAveragePooling1D()(s2)
    merge_text = multiply([s1, s2])
    x = Dense(100, activation='linear')(merge_text)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    x = Dense(int((hidden_dim+7)/2), activation='linear')(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    main_output = Dense(2, activation='softmax')(x)
    merge_model = Model(inputs=[seq_input1, seq_input2], outputs=[main_output])
    return merge_model

batch_size1 = 256

#copy below
num_hit = 0.
num_total = 0.
num_pos = 0.
num_true_pos = 0.
num_false_pos = 0.
num_true_neg = 0.
num_false_neg = 0.

f_out = open(out_file, 'w')

for fold_count in range(5):
    model_name = rst_file.split('.')[0] + '_weights_fold{}.h5'.format(fold_count)
    if not os.path.exists(model_name):
        continue
    merge_model = None
    merge_model = build_model()
    merge_model.load_weights(model_name) #
    print('==============================')
    print(f'Loaded {model_name}')

    test_gen = DataGenerator(emb_h5, seq_array, seq_index1, seq_index2,
                             seq_size, batch_size1, shuffle=False)

    # pred = merge_model.predict([seq_tensor[seq_index1], seq_tensor[seq_index2]])
    pred = merge_model.predict_generator(test_gen)
    for i in range(len(class_labels)):
        num_total += 1
        if np.argmax(class_labels[i]) == np.argmax(pred[i]):
            num_hit += 1
        if class_labels[i][0] > 0.:
            num_pos += 1.
            if pred[i][0] > pred[i][1]:
                num_true_pos += 1
            else:
                num_false_neg += 1
        else:
            if pred[i][0] > pred[i][1]:
                num_false_pos += 1
            else:
                num_true_neg += 1

    # import pdb; pdb.set_trace()

    accuracy = num_hit / num_total
    prec = num_true_pos / (num_true_pos + num_false_pos)
    recall = num_true_pos / num_pos
    spec = num_true_neg / (num_true_neg + num_false_neg)
    f1 = 2. * prec * recall / (prec + recall)
    mcc = (num_true_pos * num_true_neg - num_false_pos * num_false_neg) / ((num_true_pos + num_true_neg) * (num_true_pos + num_false_neg) * (num_false_pos + num_true_neg) * (num_false_pos + num_false_neg)) ** 0.5
    print (accuracy, prec, recall, spec, f1, mcc)

    aupr = average_precision_score(class_labels[:,0], pred[:,0])
    auroc = roc_auc_score(class_labels[:,0], pred[:,0])
    print('aupr: {} \t auroc: {} '.format(aupr, auroc))

    f_out.write('fold: {} \n'.format(fold_count))
    f_out.write('acc=' + str(accuracy) + '\tprec=' + str(prec) + '\trecall=' + str(recall) + '\tspec=' + str(spec) + '\tf1=' + str(f1) + '\tmcc=' + str(mcc))
    f_out.write('\n')
    f_out.write('aupr: {} \t auroc: {} \n'.format(aupr, auroc))
    
accuracy = num_hit / num_total
prec = num_true_pos / (num_true_pos + num_false_pos)
recall = num_true_pos / num_pos
spec = num_true_neg / (num_true_neg + num_false_neg)
f1 = 2. * prec * recall / (prec + recall)
mcc = (num_true_pos * num_true_neg - num_false_pos * num_false_neg) / ((num_true_pos + num_true_neg) * (num_true_pos + num_false_neg) * (num_false_pos + num_true_neg) * (num_false_pos + num_false_neg)) ** 0.5
print (accuracy, prec, recall, f1)

f_out.write('overall\n')
f_out.write('acc=' + str(accuracy) + '\tprec=' + str(prec) + '\trecall=' + str(recall) + '\tspec=' + str(spec) + '\tf1=' + str(f1) + '\tmcc=' + str(mcc))


f_out.close()
