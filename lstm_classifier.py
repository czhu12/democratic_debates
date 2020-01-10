import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
import spacy
import numpy as np
from sklearn.preprocessing import LabelEncoder
nlp = spacy.load('en_core_web_sm')

def tokenizer(text): # create a tokenizer function
    return [tok.text for tok in nlp.tokenizer(text)]

class LSTM_Classifier(nn.Module):
    def __init__(self, vocab, dim_embedding, dim_hidden_size, dim_classes):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(len(vocab), dim_embedding)
        self.rnn = nn.LSTM(dim_embedding, dim_hidden_size, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(dim_hidden_size, dim_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.init_weights()

    def init_weights(self):
        for params in self.rnn.parameters():
            pass

    def forward(self, inputs, target=None):
        output = {}
        embedded = self.embedding(inputs)
        sequenced, (final_hidden_state, final_cell_state) = self.rnn(embedded)
        prediction = self.linear(final_hidden_state[0])
        output['prediction'] = prediction

        if target is not None:
            import pdb; pdb.set_trace()
            loss = self.criterion(prediction, target)
            output['loss'] = loss
            output['correct'] = torch.max(prediction, 1)[1] == target
        return output



class LSTMClassifier(nn.Module):
    def __init__(self, output_size, hidden_size, vocab_size, embedding_length):
        super(LSTMClassifier, self).__init__()
        """
        Arguments
        ---------
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table

        """

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)# Initializing the look-up table.
        self.lstm = nn.LSTM(embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    def forward(self, input_sentence):
        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
        final_output.shape = (batch_size, output_size)

        """

        ''' Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins.'''
        inputs = self.word_embeddings(input_sentence)
        inputs = inputs.permute(1, 0, 2)
        output, (final_hidden_state, final_cell_state) = self.lstm(inputs)
        final_output = self.label(final_hidden_state[-1])
        return final_output


def main():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    all_df = train_df.append(test_df)
    text_encoder = data.Field(sequential=True, tokenize=tokenizer, lower=True, pad_first=True)
    text_encoder.build_vocab(all_df.text.str.lower().apply(tokenizer), vectors="glove.6B.100d")
    output = LabelEncoder().fit_transform(all_df.class_name)
    dim_classes = len(np.unique(output))
    #lstm_classifier = LSTMClassifier(
    #    text_encoder.vocab,
    #    dim_embedding=100,
    #    dim_hidden_size=128,
    #    dim_classes=dim_classes,
    #)
    lstm_classifier = LSTMClassifier(
        vocab_size=len(text_encoder.vocab),
        embedding_length=100,
        hidden_size=128,
        output_size=dim_classes,
    )

    epochs = 100
    batch_size = 32
    num_batches = len(train_df) // batch_size
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(lstm_classifier.parameters(), lr=0.2, momentum=0.98)
    for epoch in range(epochs):
        running_loss = 0.
        for b in range(num_batches):
            batch_df = train_df.iloc[b * batch_size : (b + 1) * batch_size]
            inputs = torch.transpose(text_encoder.process(batch_df.text), 0, 1)
            target = torch.tensor(output[b * batch_size : (b + 1) * batch_size])

            output = lstm_classifier(inputs)

            import pdb; pdb.set_trace()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("Epoch: {} | Loss: {} | Accuracy: {}".format(
            epoch,
            running_loss,
        ))


if __name__ == "__main__":
    main()
