import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sig = nn.Sigmoid()
        
        self.word_dict = None

    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        x = x.t() #transpose tensor
        lengths = x[0,:]  #after transpose, the second dimension is the length (0 got rid of the first dimension)
        reviews = x[1:,:] #removing the first line from the first dimension i.e. [501,50 -> [500,50]] - file concat on axis 1, first value is label, second value is length, third value till the end is review. Torch utils has already separated out the y label so remaining x is 501.
        embeds = self.embedding(reviews)
        lstm_out, _ = self.lstm(embeds)
        out = self.dense(lstm_out)
        out = out[lengths - 1, range(len(lengths))] #reshaping the output to be batch size?
        return self.sig(out.squeeze()) #?