import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

from CreateDataLoader import *

max_vocab_size = 20000
BATCH_SIZE = 32
emb_dim = 200
learning_rate = 0.0005
num_epochs = 10
"""
Loading data
"""
train_target = pkl.load(open("train_target.p","rb"))
val_target = pkl.load(open("val_target.p", "rb"))
test_target = pkl.load(open("test_target.p", "rb"))

train_data_tokens1 = pkl.load(open("train_data_tokens1.p", "rb"))
all_train_tokens1 = pkl.load(open("all_train_tokens1.p", "rb"))
val_data_tokens1 = pkl.load(open("val_data_tokens1.p", "rb"))
test_data_tokens1 = pkl.load(open("test_data_tokens1.p", "rb"))

train_data_tokens2 = pkl.load(open("train_data_tokens2.p", "rb"))
all_train_tokens2 = pkl.load(open("all_train_tokens2.p", "rb"))
val_data_tokens2 = pkl.load(open("val_data_tokens2.p", "rb"))
test_data_tokens2 = pkl.load(open("test_data_tokens2.p", "rb"))
#
# train_data_tokens3 = pkl.load(open("train_tokens_NP3.p", "rb"))
# all_train_tokens3 = pkl.load(open("all_train_tokens_NP3.p", "rb"))
# val_data_tokens3 = pkl.load(open("val_tokens_NP3.p", "rb"))
# test_data_tokens3 = pkl.load(open("test_tokens_NP3.p", "rb"))

# train_data_tokens4 = pkl.load(open("train_tokens_NP4.p", "rb"))
# all_train_tokens4 = pkl.load(open("all_train_tokens_NP4.p", "rb"))
# val_data_tokens4 = pkl.load(open("val_tokens_NP4.p", "rb"))
# test_data_tokens4 = pkl.load(open("test_tokens_NP4.p", "rb"))

train_data_tokens = list(map(lambda x: x[0]+x[1], zip(train_data_tokens1, train_data_tokens2)))
val_data_tokens = list(map(lambda x: x[0]+x[1], zip(val_data_tokens1, val_data_tokens2)))
test_data_tokens = list(map(lambda x: x[0]+x[1], zip(test_data_tokens1, test_data_tokens2)))

all_train_tokens = all_train_tokens1 + all_train_tokens2

# train_data_tokens = train_data_tokens4
# val_data_tokens = val_data_tokens4
# test_data_tokens = test_data_tokens4
# all_train_tokens = all_train_tokens4

print("Total number of tokens in train dataset is ", len(all_train_tokens))

"""
Building Vocabulary
"""

token2id, id2token, count_barrier = build_vocab(all_train_tokens, max_vocab_size)
print("Completing building vocabulary.")
print("The counts of words in vocabulary is larger than {}".format(count_barrier))

"""
Convert token to id in the dataset
"""
train_data_indices = token2index_dataset(train_data_tokens, token2id)
val_data_indices = token2index_dataset(val_data_tokens, token2id)
test_data_indices = token2index_dataset(test_data_tokens, token2id)
"""
Create PyTorch DataLoader
"""
train_dataset = ReviewsDataset(train_data_indices, train_target)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=review_collate_func,
                                           shuffle=True)

val_dataset = ReviewsDataset(val_data_indices, val_target)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=review_collate_func,
                                           shuffle=True)

test_dataset = ReviewsDataset(test_data_indices, test_target)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=review_collate_func,
                                           shuffle=False)
"""
Build model
"""
class BagOfWords(nn.Module):
    """
    BagOfWords classification model
    """
    def __init__(self, vocab_size, emb_dim):
        """
        @param vocab_size: size of the vocabulary.
        @param emb_dim: size of the word embedding
        """
        super(BagOfWords, self).__init__()
        # pay attention to padding_idx
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.linear = nn.Linear(emb_dim,20)

    def forward(self, data, length):
        """

        @param data: matrix of size (batch_size, max_sentence_length). Each row in data represents a
            review that is represented using n-gram index. Note that they are padded to have same length.
        @param length: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data.
        """
        out = self.embed(data)
        out = torch.sum(out, dim=1)
        out /= length.view(length.size()[0],1).expand_as(out).float()

        # return logits
        out = self.linear(out.float())
        return out

model = BagOfWords(len(id2token), emb_dim)

"""
Criterion and Optimizer
"""
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

"""
============================================================================
log linear decay learning rate
"""
# from torch.optim.lr_scheduler import LambdaLR
# lambda1 = lambda epoch: 1/(epoch+1)
# scheduler = LambdaLR(optimizer, lr_lambda=lambda1)
"""
===========================================================================
"""

"""
Train model
"""
def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    model.eval()
    for data, lengths, labels in loader:
        data_batch, length_batch, label_batch = data, lengths, labels
        outputs = F.softmax(model(data_batch, length_batch), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]

        total += labels.size(0)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()
    return (100 * correct / total)

record_val_acc = []
step_number = []

for epoch in range(num_epochs):
    #scheduler.step()
    for i, (data, lengths, labels) in enumerate(train_loader):
        model.train()
        data_batch, length_batch, label_batch = data, lengths, labels
        optimizer.zero_grad()
        outputs = model(data_batch, length_batch)
        loss = criterion(outputs, label_batch)
        loss.backward()
        optimizer.step()
        # validate every 100 iterations
        if i > 0 and i % 100 == 0:
            # validate
            val_acc = test_model(val_loader, model)
            record_val_acc.append(val_acc)
            step_number.append(i+epoch*600)
            print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}'.format(
                       epoch+1, num_epochs, i+1, len(train_loader), val_acc))

print ("After training for {} epochs".format(num_epochs))
print ("Val Acc {}".format(test_model(val_loader, model)))
print ("Test Acc {}".format(test_model(test_loader, model)))

pkl.dump(zip(step_number, record_val_acc), open("result/val_acc_vs_step_withPunc.p", "wb"))

plt.plot(step_number, record_val_acc)
plt.xlabel("step")
plt.title("Validation accuracy vs training steps")
plt.savefig("result/val_acc_vs_step_withPunc.pdf")
plt.show()
