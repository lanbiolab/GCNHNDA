# coding: utf-8
# @email: enoche.chow@gmail.com
#
# updated: Mar. 25, 2022
# Filled non-existing raw features with non-zero after encoded from encoders

"""
Data pre-processing
##########################
"""
from logging import getLogger
from collections import Counter
import os
import pandas as pd
import numpy as np
import torch
from utils.data_utils import (ImageResize, ImagePad, image_to_tensor, load_decompress_img_from_lmdb_value)
import lmdb


class RecDataset(object):
    def __init__(self, config, df=None):
        self.config = config
        self.logger = getLogger()

        # data path & files
        self.dataset_name = config['dataset']
        # Handle relative data_path: resolve from src directory
        data_path = config['data_path']
        if not os.path.isabs(data_path):
            # Get src directory (parent of utils)
            src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(src_dir, data_path)
        self.dataset_path = os.path.abspath(data_path + self.dataset_name)

        # dataframe
        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']
        self.splitting_label = self.config['inter_splitting_label']

        if df is not None:
            self.df = df
            self.inter_num = len(df)
            return
        # if all files exists
        check_file_list = [self.config['inter_file_name']]
        for i in check_file_list:
            file_path = os.path.join(self.dataset_path, i)
            if not os.path.isfile(file_path):
                raise ValueError('File {} not exist'.format(file_path))

        # load rating file from data path?
        self.load_inter_graph(config['inter_file_name'])
        self.item_num = int(max(self.df[self.iid_field].values)) + 1
        self.user_num = int(max(self.df[self.uid_field].values)) + 1
        self.inter_num = len(self.df)

    def load_inter_graph(self, file_name):
        inter_file = os.path.join(self.dataset_path, file_name)
        cols = [self.uid_field, self.iid_field, self.splitting_label]
        self.df = pd.read_csv(inter_file, usecols=cols, sep=self.config['field_separator'])
        if not self.df.columns.isin(cols).all():
            raise ValueError('File {} lost some required columns.'.format(inter_file))

    def split(self):
        dfs = []
        # splitting into training/validation/test
        for i in range(3):
            temp_df = self.df[self.df[self.splitting_label] == i].copy()
            temp_df.drop(self.splitting_label, inplace=True, axis=1)        # no use again
            dfs.append(temp_df)
        if self.config['filter_out_cod_start_users']:
            # filtering out new users in val/test sets
            train_u = set(dfs[0][self.uid_field].values)
            for i in [1, 2]:
                dropped_inter = pd.Series(True, index=dfs[i].index)
                dropped_inter ^= dfs[i][self.uid_field].isin(train_u)
                dfs[i].drop(dfs[i].index[dropped_inter], inplace=True)

        # wrap as RecDataset
        full_ds = [self.copy(_) for _ in dfs]
        return full_ds

    def kfold_split(self, k=5, seed=42, val_ratio=0.1):
        """
        K 折交叉验证划分
        
        将数据随机打乱后划分为 K 折，每折依次作为测试集，
        其余 K-1 折作为训练集（训练集内部再划分验证集）
        
        Args:
            k: 折数，默认 5
            seed: 随机种子
            val_ratio: 验证集比例，从训练集中划分
        
        Returns:
            list: K 个元组，每个元组为 (train_df, val_df, test_df)
        """
        np.random.seed(seed)
        
        # 随机打乱数据
        indices = np.random.permutation(len(self.df))
        shuffled_df = self.df.iloc[indices].reset_index(drop=True)
        
        # 划分为 K 折
        fold_size = len(self.df) // k
        folds = []
        
        for i in range(k):
            start_idx = i * fold_size
            if i == k - 1:
                # 最后一折包含剩余所有数据
                end_idx = len(self.df)
            else:
                end_idx = start_idx + fold_size
            
            fold_df = shuffled_df.iloc[start_idx:end_idx].copy()
            folds.append(fold_df)
        
        # 生成 K 个训练/验证/测试划分
        kfold_splits = []
        for i in range(k):
            # 第 i 折作为测试集
            test_df = folds[i].copy()
            
            # 其余折合并作为训练集
            train_dfs = [folds[j] for j in range(k) if j != i]
            train_df = pd.concat(train_dfs, ignore_index=True)
            
            # 从训练集中划分验证集
            val_size = int(len(train_df) * val_ratio)
            train_df_shuffled = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
            val_df = train_df_shuffled.iloc[:val_size].copy()
            train_df_final = train_df_shuffled.iloc[val_size:].copy()
            
            kfold_splits.append((train_df_final, val_df, test_df))
        
        return kfold_splits

    def copy(self, new_df):
        """Given a new interaction feature, return a new :class:`Dataset` object,
                whose interaction feature is updated with ``new_df``, and all the other attributes the same.

                Args:
                    new_df (pandas.DataFrame): The new interaction feature need to be updated.

                Returns:
                    :class:`~Dataset`: the new :class:`~Dataset` object, whose interaction feature has been updated.
                """
        nxt = RecDataset(self.config, new_df)

        nxt.item_num = self.item_num
        nxt.user_num = self.user_num
        nxt.inter_num = len(new_df)
        return nxt

    def get_user_num(self):
        return self.user_num

    def get_item_num(self):
        return self.item_num

    def shuffle(self):
        """Shuffle the interaction records inplace.
        """
        self.df = self.df.sample(frac=1, replace=False).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Series result
        return self.df.iloc[idx]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = [self.dataset_name]
        self.inter_num = len(self.df)
        uni_u = pd.unique(self.df[self.uid_field])
        uni_i = pd.unique(self.df[self.iid_field])
        tmp_user_num, tmp_item_num = 0, 0
        if self.uid_field:
            tmp_user_num = len(uni_u)
            avg_actions_of_users = self.inter_num/tmp_user_num
            info.extend(['The number of users: {}'.format(tmp_user_num),
                         'Average actions of users: {}'.format(avg_actions_of_users)])
        if self.iid_field:
            tmp_item_num = len(uni_i)
            avg_actions_of_items = self.inter_num/tmp_item_num
            info.extend(['The number of items: {}'.format(tmp_item_num),
                         'Average actions of items: {}'.format(avg_actions_of_items)])
        info.append('The number of inters: {}'.format(self.inter_num))
        if self.uid_field and self.iid_field:
            sparsity = 1 - self.inter_num / tmp_user_num / tmp_item_num
            info.append('The sparsity of the dataset: {}%'.format(sparsity * 100))
        return '\n'.join(info)
