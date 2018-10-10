from os import listdir
from nltk import ngrams
import string
import re
import pickle as pkl

"""
Read data from files
"""
def Read_DataFile(DirName):
    data = []
    for f in listdir(DirName):
        if f.endswith('.txt'):
            with open(DirName+'/'+f,'r') as file:
                text = file.read()
                data.append(text)
    return data

train_pos = Read_DataFile("aclImdb/train/pos")
train_neg = Read_DataFile("aclImdb/train/neg")
test_pos = Read_DataFile("aclImdb/test/pos")
test_neg = Read_DataFile("aclImdb/test/neg")

"""
Split train data and validation data
"""
train_split = 10000

train_data = train_pos[:train_split] + train_neg[:train_split]
train_target = [1]*len(train_pos[:train_split]) + [0]*len(train_neg[:train_split])

val_data = train_pos[train_split:] + train_neg[train_split:]
val_target = [1]*len(train_pos[train_split:]) + [0]*len(train_neg[train_split:])

test_data = test_pos + test_neg
test_target = [1]*len(test_pos) + [0]*len(test_neg)

print("Train dataset size is {}".format(len(train_target)))
print("Val dataset size is {}".format(len(val_target)))
print("Test dataset size is {}".format(len(test_target)))

pkl.dump(train_target, open("train_target.p", "wb"))
pkl.dump(val_target, open("val_target.p", "wb"))
pkl.dump(test_target, open("test_target.p", "wb"))

"""
Tokenize data
"""
def tokenize_ngrams(n, sent):
    #tokens = ngrams(re.findall(r"[\w']+|[.,!?;():~@+-<>#]", sent.lower()),n)
    tokens = ngrams([gram for gram in re.findall(r"[\w']+|[.,!?;():~@+-<>#]", sent.lower())
                     if (gram not in string.punctuation)], n)
    return [token for token in tokens]

def tokenize_dataset_ngrams(n, dataset, if_all_tokens = False):
    if if_all_tokens:
        token_dataset = []
        all_tokens = []

        for sample in dataset:
            tokens = tokenize_ngrams(n, sample)
            token_dataset.append(tokens)
            all_tokens += tokens

        return token_dataset, all_tokens

    else:
        token_dataset = []
        for sample in dataset:
            tokens = tokenize_ngrams(n, sample)
            token_dataset.append(tokens)
        return token_dataset

print("Saving tokenized data without punctuations")
for n in range(1,5):
    print(n)
    print ("Tokenizing val data")
    val_data_tokens = tokenize_dataset_ngrams(n, val_data)
    pkl.dump(val_data_tokens, open("val_tokens_NP"+str(n)+".p", "wb"))

    # test set tokens
    print ("Tokenizing test data")
    test_data_tokens = tokenize_dataset_ngrams(n, test_data)
    pkl.dump(test_data_tokens, open("test_tokens_NP"+str(n)+".p", "wb"))

    # train set tokens
    print ("Tokenizing train data")
    train_data_tokens, all_train_tokens = tokenize_dataset_ngrams(n, train_data, if_all_tokens = True)
    pkl.dump(train_data_tokens, open("train_tokens_NP"+str(n)+".p", "wb"))
    pkl.dump(all_train_tokens, open("all_train_tokens_NP"+str(n)+".p", "wb"))
