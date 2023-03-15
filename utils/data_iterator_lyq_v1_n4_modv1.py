import numpy as np
import random
import pickle as pkl
import gzip
import cv2
import lmdb
from torch.utils.data import Dataset, DataLoader
import torch
import sys

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

class dataIterator(Dataset):
    def __init__(self, feature_file, label_file, align_file):
        fp_feature=open(feature_file,'rb')
        self.features=pkl.load(fp_feature)
        fp_feature.close()

        fp_label=open(label_file,'rb')
        self.labels=pkl.load(fp_label)
        fp_label.close()

        fp_align=open(align_file,'rb')
        self.aligns=pkl.load(fp_align)
        fp_align.close()

    def __len__(self):
        return len(list(self.labels.keys()))
    
    def __getitem__(self, index):
        idx_key = list(self.labels.keys())[index]
        image = self.features[idx_key]
        label = self.labels[idx_key]
        align = self.aligns[idx_key]
        return image, label, align

class data_preprocessing(object):
    def __init__(self, diction_object, diction_relation, diction_relation2):
        self.diction_object = diction_object
        self.diction_relation = diction_relation
        self.diction_relation2 = diction_relation2

    def __call__(self, batch):
        diction_object = self.diction_object
        diction_relation = self.diction_relation
        diction_relation2 = self.diction_relation2
        images, labels, labels2 = zip(*batch)
        n_samples = len(labels)
        h, w = images[0].shape
        # image preprocess
        new_images = []
        for img in images:
            img = img[None, :, :] / 255.
            new_images.append(torch.from_numpy(img))
        new_images = torch.cat([img.unsqueeze(0) for img in new_images], 0)
        new_image_masks = torch.ones((n_samples, h, w))
        # label preprocess
        len_labels_l = []
        len_labels_r = []
        ltargets = []
        rtargets = []
        lpositions = []
        rpositions = []
        relations = []
        relations2 = []
        for label2 in labels2:
            relation2_list = []
            for line_idx, line in enumerate(label2):
                relation2 = line[4]
                if line_idx != len(label2)-1:
                    if diction_relation2.__contains__(relation2):
                        relation2_list.append(diction_relation2[relation2])
                else:
                    relation2_list.append(0) # whatever which one to replace End relation
            relations2.append(relation2_list)
             
        for label in labels:    # batch
            lchar_list = []
            rchar_list = []
            lpos_list = []
            rpos_list = []
            relation_list = []
            for line_idx, line in enumerate(label): # single sample
                parts = line
                lchar = parts[0]
                lpos = parts[1]
                rchar = parts[2]
                rpos = parts[3]
                relation = parts[4]
                # print(relation)
                if diction_object.__contains__(lchar):
                    lchar_list.append(diction_object[lchar])
                else:
                    print ('a symbol not in the dictionary !! formula',line_idx ,'symbol', lchar)
                    sys.exit()
                if diction_object.__contains__(rchar):
                    rchar_list.append(diction_object[rchar])
                else:
                    print ('a symbol not in the dictionary !! formula',line_idx ,'symbol', rchar)
                    sys.exit()
                
                lpos_list.append(int(lpos))
                rpos_list.append(int(rpos))  

                if line_idx != len(label)-1:
                    if diction_relation.__contains__(relation):
                        relation_list.append(diction_relation[relation])
                else:
                    relation_list.append(0) # whatever which one to replace End relation
            
            # import pdb; pdb.set_trace()
            length_l = len(lchar_list)
            length_r = len(rchar_list)
            len_labels_l.append(length_l)
            len_labels_r.append(length_r)
            ltargets.append(lchar_list)
            rtargets.append(rchar_list)
            lpositions.append(lpos_list)
            rpositions.append(rpos_list) 
            relations.append(relation_list)


        maxlen_ly = max(len_labels_l)
        maxlen_ry = max(len_labels_r)
        ly = np.zeros((maxlen_ly, n_samples)).astype(np.int64)  # <eos> must be 0 in the dict
        ry = np.zeros((maxlen_ry, n_samples)).astype(np.int64)
        re = np.zeros((maxlen_ly, n_samples)).astype(np.int64)
        re2 = np.zeros((maxlen_ly, n_samples)).astype(np.int64)
        ma = np.zeros((n_samples, maxlen_ly, maxlen_ly)).astype(np.int64)
        lp = np.zeros((maxlen_ly, n_samples)).astype(np.int64)
        rp = np.zeros((maxlen_ry, n_samples)).astype(np.int64) 
        ly_mask = np.zeros((maxlen_ly, n_samples)).astype(np.float32)
        ry_mask = np.zeros((maxlen_ry, n_samples)).astype(np.float32)
        re_mask = np.zeros((maxlen_ly, n_samples)).astype(np.float32)
        re2_mask = np.zeros((maxlen_ly, n_samples)).astype(np.float32)
        ma_mask = np.zeros((n_samples, maxlen_ly, maxlen_ly)).astype(np.float32) 

        for idx, (lchar_list, rchar_list, lpos_list, rpos_list, relation2_list, relation_list) in enumerate(zip(ltargets, rtargets, lpositions, rpositions, relations2, relations)):

            ly[:len_labels_l[idx], idx] = lchar_list
            ry[:len_labels_r[idx], idx] = rchar_list
            re[:len_labels_l[idx], idx] = relation_list
            re2[:len_labels_l[idx], idx] = relation2_list
            lp[:len_labels_l[idx], idx] = lpos_list
            rp[:len_labels_r[idx], idx] = rpos_list

            # ma[idx, :len_labels_l[idx], :len_labels_l[idx]] = align
            ly_mask[:len_labels_l[idx]] = 1.
            ry_mask[:len_labels_r[idx]] = 1.
            ry_mask[0, idx] = 0. # remove the <s>
            re_mask[:len_labels_l[idx]] = 1.
            re2_mask[:len_labels_l[idx]] = 1.
            re_mask[0, idx] = 0. # remove the Start relation
            re_mask[len_labels_l[idx]-1, idx] = 0. # remove the End relation   

        return new_images, new_image_masks, ly, ly_mask, ry, ry_mask, lp, rp, re, re2, re2_mask

