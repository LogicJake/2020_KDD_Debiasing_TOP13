import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.layers import *
from keras import backend as BKD
import keras.backend.tensorflow_backend as TFBKD
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
import gensim
import gc
import time
pd.set_option('display.max_columns', None)
tqdm.pandas(desc='pandas bar')

import logging
logging.basicConfig(
    level=logging.DEBUG,
    filename='logging_log.txt',
    filemode='w',
    format= '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
)


class TargetAttentionLayer(Layer):
    def __init__(self, n_steps, embedding_size, hidden_dim, random_state, **kwargs):
        self.n_steps = n_steps
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim
        self.random_state = random_state
        super(TargetAttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='W',
            shape=(self.embedding_size, self.hidden_dim),
            initializer=keras.initializers.truncated_normal(stddev=0.1, seed=self.random_state),
            trainable=True
        )
        self.W_BIAS = self.add_weight(
            name='W_BIAS',
            shape=(self.hidden_dim,),
            initializer=keras.initializers.Constant(value=0.1),
            trainable=True
        )
        self.O = self.add_weight(
            name='O',
            shape=(self.hidden_dim, 1),
            initializer=keras.initializers.truncated_normal(stddev=0.1, seed=self.random_state),
            trainable=True
        )
        self.O_BIAS = self.add_weight(
            name='O_BIAS',
            shape=(1,),
            initializer=keras.initializers.Constant(value=0.1),
            trainable=True
        )
        super(TargetAttentionLayer, self).build(input_shape)
    
    def call(self, x):
        his_embedding_vectors, target_embedding_vector, mask = x
        mask = BKD.reshape(mask, [-1, 1, self.n_steps])
        target_embedding_vector = BKD.tile(target_embedding_vector, [1, self.n_steps, 1])
        diff = his_embedding_vectors - target_embedding_vector
        attention_vec = concatenate([his_embedding_vectors, diff, target_embedding_vector], axis=2)
        alpha = BKD.relu(BKD.bias_add(BKD.dot(attention_vec, self.W), self.W_BIAS))
        alpha = BKD.bias_add(BKD.dot(alpha, self.O), self.O_BIAS)
        alpha = BKD.permute_dimensions(alpha, [0, 2, 1]) - mask * 1e10
        attention = BKD.batch_dot(BKD.softmax(alpha), his_embedding_vectors)
        return BKD.squeeze(attention, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])


class SelfAttentionLayer(Layer):
    def __init__(self, head_count, n_steps, embedding_size, QKV_dim, random_state, **kwargs):
        self.head_count = head_count
        self.n_steps = n_steps
        self.embedding_size = embedding_size
        self.QKV_dim = QKV_dim
        self.random_state = random_state
        super(SelfAttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.WQ, self.WK, self.WV = [], [], []
        for i in range(self.head_count):
            self.WQ.append(
                self.add_weight(
                    name='WQ{}'.format(i),
                    shape=(self.embedding_size, self.QKV_dim),
                    initializer=keras.initializers.truncated_normal(stddev=0.1, seed=self.random_state),
                    trainable=True
                )
            )
            self.WK.append(
                self.add_weight(
                    name='WK{}'.format(i),
                    shape=(self.embedding_size, self.QKV_dim),
                    initializer=keras.initializers.truncated_normal(stddev=0.1, seed=self.random_state),
                    trainable=True
                )
            )
            self.WV.append(
                self.add_weight(
                    name='WV{}'.format(i),
                    shape=(self.embedding_size, self.QKV_dim),
                    initializer=keras.initializers.truncated_normal(stddev=0.1, seed=self.random_state),
                    trainable=True
                )
            )
        self.BE = self.add_weight(
            name='BE',
            shape=(self.n_steps, self.embedding_size),
            initializer=keras.initializers.truncated_normal(stddev=0.1, seed=self.random_state),
            trainable=True
        )
        self.WO = self.add_weight(
            name='WO',
            shape=(self.head_count * self.QKV_dim, self.embedding_size),
            initializer=keras.initializers.truncated_normal(stddev=0.1, seed=self.random_state),
            trainable=True
        )
        super(SelfAttentionLayer, self).build(input_shape)
    
    def call(self, x):
        his_embedding_vectors, mask = x
        mask = BKD.reshape(mask, [-1, 1, self.n_steps])
        mask = BKD.tile(mask, [1, self.n_steps, 1])
        his_embedding_vectors = his_embedding_vectors + self.BE
        attention = []
        for i in range(self.head_count):
            Q = BKD.dot(his_embedding_vectors, self.WQ[i])
            K = BKD.dot(his_embedding_vectors, self.WK[i])
            V = BKD.dot(his_embedding_vectors, self.WV[i])
            alpha = BKD.batch_dot(Q, BKD.permute_dimensions(K, [0, 2, 1])) / np.sqrt(self.embedding_size)
            alpha = BKD.softmax(alpha - mask * 1e10)
            attention.append(BKD.batch_dot(alpha, V))
        if self.head_count > 1:
            attention = concatenate(attention, axis=2)
        else:
            attention = attention[0]
        return BKD.dot(attention, self.WO)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2])


def TargetAttentionNet(
    n_steps, vocab_size, embedding_matrix_init, embedding_size, feats_dim,
    head_count, QKV_dim, attention_hidden_dim=32, output_dim=2, random_state=None
):
    X_HIS = Input(shape=(n_steps,), dtype='int32')
    X_MASK = Input(shape=(n_steps,), dtype='float32')
    X_TARGET = Input(shape=(1,), dtype='int32')
    X_FEATS = Input(shape=(feats_dim,), dtype='float32')
    
    embedding_matrix = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_size,
        embeddings_initializer=keras.initializers.Constant(value=embedding_matrix_init),
        dtype='float32',
        trainable=False
    )
    
    his_embedding_vectors = embedding_matrix(X_HIS)
    his_embedding_vectors = SelfAttentionLayer(head_count, n_steps, embedding_size, QKV_dim, random_state)([his_embedding_vectors, X_MASK])
    lstm_out = Bidirectional(CuDNNLSTM(embedding_size, return_sequences=True), merge_mode='sum')(his_embedding_vectors)
    
    target_embedding_vector = embedding_matrix(X_TARGET)
    attention_vector = TargetAttentionLayer(
        n_steps, 3 * embedding_size, attention_hidden_dim, random_state, name='emb')([lstm_out, target_embedding_vector, X_MASK])
    target_embedding_vector = Lambda(lambda x: BKD.reshape(x, [-1, embedding_size]))(target_embedding_vector)
    
    dense_input = concatenate([
        attention_vector,
        target_embedding_vector,
        X_FEATS
    ], axis=1)
        
    a = Dense(
        512,
        activation='relu',
        kernel_initializer=keras.initializers.truncated_normal(stddev=0.1, seed=random_state),
        bias_initializer=keras.initializers.Constant(value=0.1)
    )(dense_input)
    a = Dense(
        128,
        activation='relu',
        kernel_initializer=keras.initializers.truncated_normal(stddev=0.1, seed=random_state),
        bias_initializer=keras.initializers.Constant(value=0.1)
    )(a)
    output = Dense(
        output_dim,
        activation='softmax',
        kernel_initializer=keras.initializers.truncated_normal(stddev=0.1, seed=random_state),
        bias_initializer=keras.initializers.Constant(value=0.1)
    )(a)
    
    model = keras.models.Model(inputs=[X_HIS, X_MASK, X_TARGET, X_FEATS], outputs=output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
    )
    model.summary()
    return model


df = pd.read_pickle('../user_data/data/nn_input.pkl')
print(df.shape)


oof_df = df[~df['label'].isna()].reset_index(drop=True)[['user_id', 'phase', 'item_id']]
test_pred_df = df[df['label'].isna()].reset_index(drop=True)[['user_id', 'phase', 'item_id']]


emb_size = 32
sentences = df.drop_duplicates(['user_id', 'phase'])['history'].values.tolist()
print(len(sentences))
for i in tqdm(range(len(sentences))):
    sentences[i] = [str(x) for x in sentences[i]]
model = Word2Vec(sentences, size=emb_size, window=5, min_count=1, sg=1, hs=0, iter=5, seed=2020, workers=12)
del sentences
gc.collect()


feats_cols = [f for f in df.columns if f not in ['user_id', 'phase', 'query_time', 'item_id', 'label', 'history']]
drop_cols = []
for f in feats_cols:
    if df[f].count() / df[f].shape[0] <= 0.7:
        drop_cols.append(f)
print(drop_cols)
feats_cols = [f for f in feats_cols if f not in drop_cols]
for f in tqdm(feats_cols):
    df[f] = df[f].astype('float32')
    df[f] = df[f].fillna(df[f].mean())
    df[f] = StandardScaler().fit_transform(df[[f]].values).squeeze()
print(df.shape)


words = set()
for line in tqdm(df['history'].values):
    words |= set(line)
words = np.sort(list(words))
print(words[:10])
vocab_size = len(words)
words_idx_dict = dict(zip(words, range(vocab_size)))


df['item_id'] = df['item_id'].map(words_idx_dict)
print(df['item_id'].count() / df.shape[0])
df['item_id'] = df['item_id'].fillna(vocab_size).astype('int32')
df['item_id'] += 1
print(df['item_id'].count() / df.shape[0])


df['len'] = df['history'].progress_apply(len)
print(df['len'].describe())
print(df['len'].quantile(0.9))
print(df['len'].quantile(0.95))
max_len = int(df['len'].quantile(0.95))
del df['len']


df['history'] = df['history'].progress_apply(lambda seq: [words_idx_dict[int(x)] + 1 for x in seq])
df['history'] = df['history'].progress_apply(
    lambda seq: seq[-max_len:] if len(seq) >= max_len else seq + ([0] * (max_len - len(seq)))
)


emb_size = 32
embedding_matrix = np.zeros((vocab_size + 2, emb_size))
for w in tqdm(words_idx_dict.keys()):
    if str(w) in model:
        embedding_matrix[words_idx_dict[w] + 1] = model[str(w)]
embedding_matrix = embedding_matrix.astype(np.float32)


train_df = df[~df['label'].isna()].reset_index(drop=True)
train_df['label'] = train_df['label'].astype('int8')
test_df = df[df['label'].isna()].reset_index(drop=True)


train_his = np.array(train_df['history'].values.tolist())
train_mask = (train_his == 0).astype('float32')
train_target = train_df['item_id'].values
train_feats = train_df[feats_cols].values
labels = train_df['label'].values

test_his = np.array(test_df['history'].values.tolist())
test_mask = (test_his == 0).astype('float32')
test_target = test_df['item_id'].values
test_feats = test_df[feats_cols].values

del train_df, test_df, model
gc.collect()


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='min')
output_dim = 2
oof_df['nn_prob'] = 0
test_pred_df['nn_prob'] = 0
oof_emb = np.zeros((oof_df.shape[0], emb_size))
test_emb = np.zeros((test_pred_df.shape[0], emb_size))
for i, (trn_idx, val_idx) in enumerate(skf.split(train_his, labels)):
    print('============ {} fold ============'.format(i))
    t = time.time()
    
    trn_his, trn_mask, trn_target = train_his[trn_idx], train_mask[trn_idx], train_target[trn_idx]
    trn_feats, trn_y = train_feats[trn_idx], labels[trn_idx]
    val_his, val_mask, val_target = train_his[val_idx], train_mask[val_idx], train_target[val_idx]
    val_feats, val_y = train_feats[val_idx], labels[val_idx]
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    TFBKD.set_session(sess)
    
    clf = TargetAttentionNet(
        n_steps=max_len,
        vocab_size=vocab_size + 2,
        embedding_matrix_init=embedding_matrix,
        embedding_size=emb_size,
        feats_dim=trn_feats.shape[1],
        head_count=8,
        QKV_dim=emb_size,
        attention_hidden_dim=128,
        output_dim=output_dim,
        random_state=2020
    )
    clf.fit(
        x=[trn_his, trn_mask, trn_target, trn_feats], y=np.eye(output_dim)[trn_y],
        batch_size=1024, epochs=4,
        validation_data=([val_his, val_mask, val_target, val_feats], np.eye(output_dim)[val_y]),
        verbose=1, callbacks=[early_stopping]
    )
    oof_df.loc[val_idx, 'nn_prob'] = clf.predict([val_his, val_mask, val_target, val_feats], batch_size=2048)[:, 1]
    print('val auc:', roc_auc_score(val_y, oof_df['nn_prob'].values[val_idx]))
    test_pred_df['nn_prob'] += clf.predict([test_his, test_mask, test_target, test_feats], batch_size=2048)[:, 1] / skf.n_splits
    emb_layer = keras.models.Model(inputs=clf.input, outputs=clf.get_layer(name='emb').output)
    oof_emb[val_idx] = emb_layer.predict([val_his, val_mask, val_target, val_feats], batch_size=2048)
    test_emb += emb_layer.predict([test_his, test_mask, test_target, test_feats], batch_size=2048) / skf.n_splits
    
    del emb_layer, clf
    BKD.clear_session()
    tf.reset_default_graph()
    gc.collect()
    
    print('runtime: {}\n'.format(time.time() - t))


for i in tqdm(range(emb_size)):
    oof_df['nn_emb_{}'.format(i)] = oof_emb[:, i]
    test_pred_df['nn_emb_{}'.format(i)] = test_emb[:, i]
oof_df.to_pickle('../user_data/data/nn/nn_trn.pkl')
test_pred_df.to_pickle('../user_data/data/nn/nn_test.pkl')











