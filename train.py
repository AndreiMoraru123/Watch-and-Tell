from pycocotools.coco import COCO
from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torchvision import transforms
import sys
import math


sys.path.append('PythonAPI')
dataDir = ''
dataType = 'val2017'
annFile = '{}coco/annotations/instances_{}.json'.format(dataDir, dataType)
log_file = 'training_log.txt'

# initialize COCO api
coco = COCO(annFile)

torch.cuda.empty_cache()

batch_size = 32
vocab_threshold = 4
vocab_from_file = False
embed_size = 512
hidden_size = 512
num_epochs = 7
save_every = 1
print_every = 100

transform_train = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # ImageNet params
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=vocab_from_file)

vocab_size = len(data_loader.dataset.vocab)

encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder.to(device)
decoder.to(device)

criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

params = list(decoder.parameters()) + list(encoder.embed.parameters())

optimizer = torch.optim.Adam(params, lr=0.001)

total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)


if __name__ == '__main__':

    print('Training on: ', device)

    with open(log_file, 'w') as f:

        for epoch in range(1, num_epochs + 1):

            for i_step in tqdm(range(1, total_step + 1)):

                indices = data_loader.dataset.get_train_indices()
                sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=indices)
                data_loader.batch_sampler.sampler = sampler

                images, captions = next(iter(data_loader))
                images = images.to(device)
                captions = captions.to(device)

                encoder.zero_grad()
                decoder.zero_grad()

                features = encoder(images)
                outputs = decoder(features, captions)

                loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
                loss.backward()

                # Clip gradients: gradients are modified in place
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

                # Update weights
                optimizer.step()

                if i_step % print_every == 0:
                    f.write('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} \n'
                            .format(epoch, num_epochs, i_step, total_step, loss.item()))

            print('Epoch stats: ', epoch, loss.item())

            if epoch % save_every == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    'models', 'decoder-{}.ckpt'.format(epoch)))
                torch.save(encoder.state_dict(), os.path.join(
                    'models', 'encoder-{}.ckpt'.format(epoch)))

