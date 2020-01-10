import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

import spacy
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset

from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LabelModel
from snorkel.labeling import filter_unlabeled_dataframe
from snorkel.labeling import MajorityLabelVoter

nlp = spacy.load('en_core_web_sm')

def tokenizer(text): # create a tokenizer function
    return [tok.text for tok in nlp.tokenizer(text)]


def get_majority_vote_label(train_df, lfs, labels):
    applier = PandasLFApplier([labeling_function(name=lf.__name__)(lf) for lf in lfs])
    label_model = LabelModel(cardinality=len(labels), verbose=True)
    L_train = applier.apply(df=train_df)
    majority_model = MajorityLabelVoter(cardinality=len(labels))
    preds_train = majority_model.predict(L=L_train)

    non_abstain_idxs = np.argwhere(preds_train >= 0).flatten()
    df_filtered = train_df.iloc[non_abstain_idxs]
    probs_filtered = preds_train[non_abstain_idxs]
    return df_filtered, probs_filtered


def get_snorkel_labels(train_df, lfs, labels):
    applier = PandasLFApplier([labeling_function(name=lf.__name__)(lf) for lf in lfs])
    label_model = LabelModel(cardinality=len(labels), verbose=True)
    L_train = applier.apply(df=train_df)
    label_model.fit(L_train, n_epochs=500, lr=0.001, log_freq=100, seed=123)
    L_probs = label_model.predict_proba(L=L_train)

    df_filtered, probs_filtered = filter_unlabeled_dataframe(
        X=train_df, y=L_probs, L=L_train
    )
    return df_filtered, probs_filtered

class SnorkelDataset(Dataset):
    def __init__(self, X, y, text_encoder):
        self.X = X
        self.y = y
        self.text_encoder = text_encoder

    def collate_fn(self, batch):
        xs = [s['x'] for s in batch]
        ys = np.stack([s['y'] for s in batch])
        text = [s['text'] for s in batch]
        idxs = torch.t(self.text_encoder.process(xs))
        reveds = []
        #for idx in range(len(xs)):
        #    reved = [self.text_encoder.vocab.itos[i] for i in idxs[:, idx].numpy()]
        #    reveds.append(reved)
        #import pdb; pdb.set_trace()
        return idxs, torch.from_numpy(ys), text

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        text = self.X.iloc[idx].text
        x = self.text_encoder.preprocess(text.lower())
        y = self.y[idx]
        return { 'x': x, 'y': y, 'text': text }


class SnorkelModel(nn.Module):
    def __init__(self, embedding_dim, vocab, classes_dim, hidden_size=40):
        super(SnorkelModel, self).__init__()
        self.embedding_dim = len(vocab.vectors[0])
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(len(vocab), embedding_dim)
        self.hidden_size = hidden_size

        self.rnn = nn.LSTM(embedding_dim, hidden_size, 1, batch_first=True, dropout=0.5, bidirectional=True)

        self.decoder = nn.Linear(hidden_size * 2, classes_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.init_weights(vocab.vectors)

    #def init_weights(self, vectors):
    #    self.embed.weight.data.copy_(vectors)
    #    for m in self.modules():
    #        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
    #            for name, param in m.named_parameters():
    #                if 'weight_ih' in name:
    #                    torch.nn.init.xavier_uniform_(param.data)
    #                elif 'weight_hh' in name:
    #                    torch.nn.init.orthogonal_(param.data)
    #                elif 'bias' in name:
    #                    param.data.fill_(0)

    def init_weights(self, vectors):
        self.embed.weight.data.copy_(vectors)

    def forward(self, inputs, target=None):
        obj = {}
        x = self.embed(inputs)

        output, (final_hidden_state, final_cell_state) = self.rnn(x)
        hidden_state = final_hidden_state.view(inputs.shape[0], -1)
        prediction = self.decoder(hidden_state)

        obj['prediction'] = prediction

        if target is not None:
            #softmax = F.softmax(prediction)
            #log_softmax = torch.log(softmax)
            #obj['loss'] = -torch.sum(torch.mul(log_softmax.double(), target))
            # Approximately correct targets
            obj['loss'] = self.criterion(prediction, target)
            max_predictions = torch.max(prediction, 1)[1]
            obj['correct'] = (max_predictions == target).sum()
        return obj

class Trainer():
    def __init__(self, trainset, testset, model, lr=0.2, momentum=0.9):
        self.trainset = trainset
        self.testset = testset
        self.model = model
        self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    def train(self, epochs=20):
        self.model.train()
        losses = []
        for epoch in range(epochs):
            running_loss = 0.0
            running_correct = 0.0
            for idx, sample_batched in enumerate(self.trainset):
                self.optimizer.zero_grad()

                x, y, text = sample_batched
                output = self.model(x, y)
                #itos = self.trainset.dataset.text_encoder.vocab.itos
                #texts = []
                #for i in range(x.shape[1]):
                #    texts.append(" ".join([itos[i] for i in x[:, i].numpy() if itos[i] != "<pad>"]))
                loss = output['loss']

                loss.backward()

                self.optimizer.step()
                running_loss += loss.item()
                running_correct += output['correct'].item()
                losses.append(loss.item())
            print("Epoch: {} | Loss: {} | Accuracy: {}".format(epoch, running_loss, running_correct / len(self.trainset.dataset)))
            self.validation()
        return losses

    def validation(self):
        self.model.eval()
        for idx, sample_batched in enumerate(self.testset):
            x, y, text = sample_batched
            output = self.model(x, y)
            print("Test Loss: {} | Test Accuracy: {}".format(output['loss'].item(), output['correct'].item() / len(self.testset.dataset)))


