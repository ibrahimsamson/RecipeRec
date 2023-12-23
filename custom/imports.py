import json
import pickle
import re
import nltk
from collections import Counter
import pandas as pd
import random
import heapq
import csv
from tqdm import tqdm
import os
import numpy as np
import time
import math
# import lmdb
import gensim
import heapq

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax
import torchfile
from torch.nn import init
import dgl.function as fn
from dgl.utils import expand_as_pair
# from dgl.nn import EdgeWeightNorm

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score

 
device = ("cuda:0" if torch.cuda.is_available() else "cpu")
# parameters
Ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # top@K performance
n_test_negs = 100 # number of negative recipes for each test user
dataset_folder = '../data/'
