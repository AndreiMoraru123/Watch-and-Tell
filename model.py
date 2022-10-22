import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import torchvision.models as models


class EncoderCNN(nn.Module):

    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # ResNet-101
        resnet = models.resnet101(pretrained=True)

        # behead the last linear layer (since we're not doing classification)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

        self.fine_tune()

    def forward(self, images):
        """
            Forward propagation.
            :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
            :return: encoded images
        """

        features = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        features = features.view(features.size(0), -1)  # (batch_size, 2048*image_size/32*image_size/32)
        features = self.embed(features)  # (batch_size, embed_size)

        return features

    def fine_tune(self, fine_tune=True):
        for param in self.resnet.parameters():
            param.requires_grad_(False)


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Embedding or type(m) == nn.Conv2d:
        I.kaiming_normal_(m.weight)


class DecoderRNN(nn.Module):
    """
    RNN Decoder sprinkled with attention
    """

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.8):
        super(DecoderRNN, self).__init__()
        self.n_hidden = hidden_size
        self.n_layers = num_layers
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        if self.n_layers == 1:
            self.drop_prob = 0
        else:
            self.drop_prob = dropout


        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)

        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.n_hidden, num_layers=self.n_layers,
                            dropout=self.drop_prob, batch_first=True)

        self.linear = nn.Linear(in_features=self.n_hidden, out_features=self.vocab_size)

        self.apply(init_weights)

    def forward(self, features, captions):

        captions = captions[:, :-1]
        embeds = self.embed(captions)
        embeds = torch.cat((features.unsqueeze(1), embeds), dim=1)
        output, _ = self.lstm(embeds)
        output = self.linear(output)

        return output

    def sample(self, inputs, states=None, max_len=20):

        """
        accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len)
        :param inputs:
        :param states:
        :param max_len:
        :return: predicted sentence
        """

        predicted_sentence = []

        for i in range(max_len):
            # forward pass
            lstm_out, states = self.lstm(inputs, states)
            outputs = self.linear(lstm_out)

            # get the predicted word
            outputs = outputs.squeeze(1)
            predicted = outputs.argmax(1)

            # append to the sentence
            predicted_sentence.append(predicted.item())

            if predicted.item() == 1:  # end word
                break

            if len(predicted_sentence) == max_len:
                break

            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)

        return predicted_sentence
