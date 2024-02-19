import pandas as pd
import random
import pickle
from tqdm import tqdm
from make_datasets import make_datasets
import numpy as np
from modules import *
import tensorflow as tf
from __future__ import print_function
import sys
import os
import argparse
sys.path.append("..")

from model_BINN import BINN
from make_datasets_BINN import make_datasets
from DataInput_BINN import DataIterator
from evaluation import SortItemsbyScore,Metric_HR,Metric_MRR


class DataIterator:

    def __init__(self,
                 mode,
                 data,
                 batch_size = 128,
                 neg_sample = 1,
                 all_items = None,
                 items_usr_clicked = None,
                 shuffle = True):
        self.mode = mode
        self.data = data
        self.datasize = data.shape[0]
        self.neg_count = neg_sample
        self.batch_size = batch_size
        self.item_usr_clicked = items_usr_clicked
        self.all_items = all_items
        self.shuffle = shuffle
        self.seed = 0
        self.idx=0
        self.total_batch = round(self.datasize / float(self.batch_size))

    def __iter__(self):
        return self

    def reset(self):
        self.idx = 0
        if self.shuffle:
            self.data= self.data.sample(frac=1).reset_index(drop=True)
            self.seed = self.seed + 1
            random.seed(self.seed)

    def __next__(self):

        if self.idx  >= self.datasize:
            self.reset()
            raise StopIteration

        nums = self.batch_size
        if self.datasize - self.idx < self.batch_size:
            nums  = (self.datasize - self.idx) - (self.datasize - self.idx) % 3

        cur = self.data.iloc[self.idx:self.idx+nums]

        batch_user = cur['user'].values

        batch_seq = []
        for seq in cur['seq'].values:
            batch_seq.append(seq)

        batch_pos = []
        for t in cur['target'].values:
            batch_pos.append(t)

        batch_neg = []
        if self.mode == 'train':
            for u, seq, in zip(cur['user'],cur['seq']):
                user_item_set = set(self.all_items) - set(self.item_usr_clicked[u])
                batch_neg.append(random.sample(user_item_set,len(seq)))

        self.idx += self.batch_size

        return (batch_user, batch_seq, batch_pos, batch_neg)

if __name__ == '__main__':

    d_train, d_test, d_info = make_datasets(5, 3, 4)
    num_usr, num_item, items_usr_clicked, _, _ = d_info
    all_items = [i for i in range(num_item)]

    # Define DataIterator

    trainIterator = DataIterator('train', d_train, 21, 5,
                                 all_items, items_usr_clicked, shuffle=True)
    for epoch in range(6):
        for data in tqdm(trainIterator,desc='epoch {}'.format(epoch),total=trainIterator.total_batch):
            batch_usr, batch_seq, batch_pos, batch_neg = data

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def make_datasets(data, len_Seq = 0, len_Tag =0, len_Pred = 0):

    p = data.groupby('item')['user'].count().reset_index().rename(columns={'user':'item_count'})
    data = pd.merge(data,p,how='left',on='item')
    data = data[data['item_count'] > 5].drop(['item_count'],axis=1)

   
    item_unique = data['item'].unique().tolist()
    item_map = dict(zip(item_unique, range(1,len(item_unique) + 1)))
    item_map[0] = 0
    all_item_count = len(item_map)
    data['item'] = data['item'].apply(lambda x: item_map[x])

    user_unique = data['user'].unique().tolist()
    user_map = dict(zip(user_unique, range(1, len(user_unique) + 1)))
    user_map[0] = 0
    all_user_count = len(item_map)
    data['user'] = data['user'].apply(lambda x: user_map[x])

    
    data = data.sort_values(by=['user','timestamps']).reset_index(drop=True)

    user_sessions = data.groupby('user')['item'].apply(lambda x: x.tolist()) \
        .reset_index().rename(columns={'item': 'item_list'})

    train_users = []
    train_seqs = []
    train_targets = []

    test_users = []
    test_seqs = []
    test_targets = []

    items_usr_clicked = {}

    for index, row in user_sessions.iterrows():
        user = row['user']
        items = row['item_list']

        items_usr_clicked[user] = items

        user_train = items[:-1]

        train_seq = np.zeros([len_Seq], dtype=np.int32)
        train_pos = np.zeros([len_Seq], dtype=np.int32)

        nxt = user_train[-1]
        idx = len_Seq - 1
        for i in reversed(user_train[:-1]):
            train_seq[idx] = i
            train_pos[idx] = nxt
            nxt = i
            idx -= 1
            if idx == -1: break
        train_users.append(user)
        train_seqs.append(train_seq)
        train_targets.append(train_pos)

        test_seq = np.zeros([len_Seq], dtype=np.int32)


        user_test = items
        nxt = user_test[-1]
        test_pos = [nxt]

        idx = len_Seq - 1
        for i in reversed(user_test[:-1]):
            test_seq[idx] = i
            idx -= 1
            if idx == -1: break
        test_users.append(user)
        test_seqs.append(test_seq)
        test_targets.append(test_pos)
        
    d_train = pd.DataFrame({'user':train_users,'seq':train_seqs,'target':train_targets})

    d_test = pd.DataFrame({'user': test_users, 'seq': test_seqs, 'target': test_targets})

    d_info= (all_user_count, all_item_count, items_usr_clicked, user_map, item_map)

    return d_train,d_test,d_info

class BINN():
    def __init__(self, usernum = 100,
                 itemnum = 100,
                 emb_size = 128,
                 max_Seqlens = 50,
                 num_blocks = 1,
                 num_heads = 1,
                 dropout_rate = 0.8,
                 reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.max_len))
        self.pos = tf.placeholder(tf.int32, shape=(None, args.max_len))
        self.neg = tf.placeholder(tf.int32, shape=(None, args.max_len))
        self.usernum = usernum
        self.itemnum = itemnum
        self.reuse = reuse
        self.hidden_units = emb_size
        self.l2_emb = 1e-6
        self.maxlen = max_Seqlens
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.num_heads = num_heads


        self.loss,self.seq_emb = self.build_network(self.u,self.input_seq,self.pos,self.neg)



    def build_network(self,u,input_seq,pos,neg,is_training):
        mask = tf.expand_dims(tf.to_float(tf.not_equal(input_seq, 0)), -1)

        self.input_seq = input_seq

        with tf.variable_scope("BINN", reuse=self.reuse):
            
            self.seq, item_emb_table = embedding(input_seq,
                                                 vocab_size=self.itemnum,
                                                 num_units=self.hidden_units,
                                                 zero_pad=False,
                                                 scale=True,
                                                 l2_reg=self.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=None
                                                 )
            self.item_emb_table = item_emb_table
            
            t, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(input_seq)[1]), 0), [tf.shape(input_seq)[0], 1]),
                vocab_size=self.maxlen,
                num_units=self.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=self.l2_emb,
                scope="dec_pos",
                reuse=self.reuse,
                with_t=True
            )
            self.seq += t

            
            self.seq = tf.layers.dropout(self.seq,rate=self.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask

            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):

                    
                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   num_units=self.hidden_units,
                                                   num_heads=self.num_heads,
                                                   dropout_rate=self.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    
                    self.seq = feedforward(normalize(self.seq), num_units=[self.hidden_units, self.hidden_units],
                                           dropout_rate=self.dropout_rate, is_training=self.is_training)
                    self.seq *= mask

            self.seq = normalize(self.seq)

        pos = tf.reshape(pos, [tf.shape(input_seq)[0] * self.maxlen])
        neg = tf.reshape(neg, [tf.shape(input_seq)[0] * self.maxlen])
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)
        seq_emb = tf.reshape(self.seq, [tf.shape(input_seq)[0] * self.maxlen, self.hidden_units])

        self.seq_emb  = seq_emb

        # prediction layer
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(input_seq)[0] * self.maxlen])


        self.loss = tf.reduce_sum(
            - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
            tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)

        return self.loss, self.seq_emb


    def predict(self,item_list):
        len_item = len(item_list)
        all_index = tf.convert_to_tensor(item_list, dtype=tf.int32)
        test_item_emb = tf.nn.embedding_lookup(self.item_emb_table, all_index)
        self.test_logits = tf.matmul(self.seq_emb, tf.transpose(test_item_emb))
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], self.maxlen, len_item])
        self.test_logits = self.test_logits[:, -1, :]

        return self.test_logits

def positional_encoding(dim, sentence_length, dtype=tf.float32):

    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])

    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)

def normalize(inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs

def embedding(inputs, 
              vocab_size, 
              num_units, 
              zero_pad=True, 
              scale=True,
              l2_reg=0.0,
              scope="embedding", 
              with_t=False,
              reuse=None):
    
   
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       #initializer=tf.contrib.layers.xavier_initializer(),
                                       regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        
        if scale:
            outputs = outputs * (num_units ** 0.5) 
    if with_t: return outputs,lookup_table
    else: return outputs


def multihead_attention(queries, 
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None,
                        with_qk=False):
    
        
   
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
    
        Q = tf.layers.dense(queries, num_units, activation=None) 
        K = tf.layers.dense(keys, num_units, activation=None) 
        V = tf.layers.dense(keys, num_units, activation=None)
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) 
        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) 
        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        
        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) 
        key_masks = tf.tile(key_masks, [num_heads, 1]) 
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) 
        
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
  
        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) 
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() 
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) 
            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) 
  
        # Activation
        outputs = tf.nn.softmax(outputs) 
         
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) 
        query_masks = tf.tile(query_masks, [num_heads, 1]) 
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) 
        outputs *= query_masks 
          
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
               
        # Weighted sum
        outputs = tf.matmul(outputs, V_) 
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) 
              
        # Residual connection
        outputs += queries
 
    if with_qk: return Q,K
    else: return outputs

def feedforward(inputs, 
                num_units=[2048, 512],
                scope="multihead_attention", 
                dropout_rate=0.2,
                is_training=True,
                reuse=None):
   
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        
        # Residual connection
        outputs += inputs
        
       
    return outputs

#RUN THE MODEL
def parse_args():
    parser = argparse.ArgumentParser(description='SASRec')
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--emb_size', type=int, default=128)
    parser.add_argument('--display_step', type=int, default=256)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=3e-3)
    parser.add_argument('--num_blocks', type=float, default=2)
    parser.add_argument('--num_heads', type=float, default=1)
    parser.add_argument('--keep_prob', type=float, default=0.8)
    parser.add_argument('--l2_lambda', type=float, default=1e-6)
    return parser.parse_args()




if __name__ == '__main__':

    # Get Params
    args = parse_args()

    # make datasets

    print('==> make datasets <==')
    file_path = '../../datasets/ratings1m.dat'
    names = ['user', 'item', 'rateing', 'timestamps']
    data = pd.read_csv(file_path, header=None, sep='::', names=names)
    d_train, d_test, d_info = make_datasets(data, args.max_len)


    num_usr, num_item, items_usr_clicked, _, _ = d_info
    all_items = [i for i in range(num_item)]

    # Define DataIterator
    trainIterator = DataIterator('train',d_train, args.batch_size, args.max_len,
                                 all_items, items_usr_clicked, shuffle=True)
    testIterator = DataIterator('test',d_test, args.batch_size,  shuffle=False)

    # Define Model

    model = BINN(usernum=num_user,
                   itemnum=num_item,
                   emb_size=args.emb_size,
                   max_Seqlens=args.max_len,
                   num_blocks=args.num_blocks,
                   num_heads=args.num_heads,
                   dropout_rate=args.keep_prob)

    score_pred = model.predict(all_items)
    loss = model.loss

    # Define Optimizer
    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    train_op = optimizer.minimize(loss,global_step=global_step)

    # Training and test for every epoch
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(args.num_epochs):

            #train
            cost_list = []
            for train_input in tqdm(trainIterator,desc='epoch {}'.format(epoch),total=trainIterator.total_batch):
                batch_usr, batch_seq, batch_pos, batch_neg = train_input
                feed_dict = {model.u: batch_usr, model.input_seq: batch_seq,
                            model.pos: batch_pos, model.neg: batch_neg,
                            model.is_training :True}
                _, step, cost= sess.run([train_op, global_step, loss],feed_dict)
                cost_list.append(cost)
            mean_cost = np.mean(cost_list)

            # test
            pred_list = []
            next_list = []
            user_list = []

            if epoch % 10 != 0:
                continue

            for test_input in testIterator:
                batch_usr, batch_seq, batch_pos, batch_neg = test_input
                feed_dict = {model.u: batch_usr, model.input_seq: batch_seq, model.is_training: False}
                pred = sess.run(score_pred, feed_dict)  
                pred_list += pred.tolist()
                next_list += list(batch_pos)
                user_list += list(batch_usr)


            sorted_items,sorted_score = SortItemsbyScore(all_items,pred_list,remove_hist=True
            ,reverse = True,usr=user_list,usrclick=items_usr_clicked)


            hr50 = Metric_HR(50, next_list,sorted_items)
            Mrr = Metric_MRR(50,next_list,sorted_items)

            print(" epoch {}, mean_loss{:g}, test HR@50: {:g} MRR: {:g}"
                .format(epoch + 1, mean_cost, hr50, Mrr))




