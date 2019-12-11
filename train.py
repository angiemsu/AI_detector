import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix

"""
PyTorch script finetuning a ResNet model
command line to run: python train.py --train_dir ~/train--val_dir ~/vali --test_dir ~/test  --num_workers 6 --num_epochs1 40 --num_epochs2 40 --use_gpu

"""
# Add arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='~/train')
parser.add_argument('--val_dir', default='~/vali')
parser.add_argument('--test_dir', default='~/test')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=6, type=int)
parser.add_argument('--num_epochs1', default=40, type=int)
parser.add_argument('--num_epochs2', default=40, type=int)
parser.add_argument('--use_gpu', action='store_true')

# Values to normalize images
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main(args):
  # Enable CUDA if running on GPU
  dtype = torch.FloatTensor
  if args.use_gpu:
    dtype = torch.cuda.FloatTensor

  # Transforms for data augmentation
  train_transform = T.Compose([
    T.Scale(224),
    T.CenterCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ])

  dataset = args.train_dir

  # Attempt to shuttle train/vali/test - WIP
  # batch_size = 16
  # validation_split = 0.2
  # shuffle_dataset = True
  # random_seed = 42

  #dataset_size = len(dataset)

  #train_dset = ImageFolder(dataset, transform=train_transform)
  #val_dset = ImageFolder(dataset, transform=train_transform)
  #indices = list(range(dataset_size))
  #split = int(np.floor(validation_split * dataset_size))
  #if shuffle_dataset :
   #     np.random.seed(random_seed)
   #     np.random.shuffle(indices)

  #train_indices, val_indices = indices[split:], indices[:split]
  #train_sampler = SubsetRandomSampler(train_indices)
  #valid_sampler = SubsetRandomSampler(val_indices)
  #train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
  #                                         sampler=train_sampler)
  #validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
  #                                              sampler=valid_sampler)
  #print(train_loader)

  # Custom data loader for images
  train_dset = ImageFolder(args.train_dir, transform=train_transform)
  train_loader = DataLoader(train_dset,
                     batch_size=args.batch_size,
                     num_workers=args.num_workers,
                     shuffle=True)


  val_dset = ImageFolder(args.val_dir, transform=train_transform)
  val_loader = DataLoader(val_dset,
                   batch_size=args.batch_size,
                   num_workers=args.num_workers)
  test_dset = ImageFolder(args.test_dir, transform=train_transform)
  test_loader = DataLoader(test_dset, batch_size=args.batch_size, num_workers=args.num_workers)

  # Initiaze model with parameters
  model = torchvision.models.resnet18(pretrained=True)

  num_classes = len(train_dset.classes)
  class_names =train_dset.classes
  print(class_names)
  model.fc = nn.Linear(model.fc.in_features, num_classes)

  model.type(dtype)
  loss_fn = nn.CrossEntropyLoss().type(dtype)

  for param in model.parameters():
    param.requires_grad = False
  for param in model.fc.parameters():
    param.requires_grad = True

  optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

  # Train model with epochs
  for epoch in range(args.num_epochs1):
    print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs1))
    run_epoch(model, loss_fn, train_loader, optimizer, dtype)

    train_acc = check_accuracy(model, train_loader, dtype)
    val_acc = check_accuracy(model, val_loader, dtype)

    print('Train accuracy: ', train_acc)
    print('Val accuracy: ', val_acc)
    print()

  for param in model.parameters():
    param.requires_grad = True

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

  for epoch in range(args.num_epochs2):
    print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs2))
    run_epoch(model, loss_fn, train_loader, optimizer, dtype)

    train_acc = check_accuracy(model, train_loader, dtype)
    val_acc = check_accuracy(model, val_loader, dtype)
    print('Train accuracy: ', train_acc)
    print('Val accuracy: ', val_acc)
    print()

  # Save model to disk
  model.class_to_idx = train_dset.class_to_idx
  model.cpu()
  torch.save({'arch':'resnet',
              'state_dict':model.state_dict(),
              'class_to_idx':model.class_to_idx},
              'classifier.pth')

  test_acc = check_accuracy(model, test_loader, dtype)
  print('Test accuracy :',test_acc)

# Train model for one epoch
def run_epoch(model, loss_fn, loader, optimizer, dtype):
  model.train()
  model.cuda()

  # One epoch of training, calculate loss and back propagate
  for x, y in loader:
    x_var = Variable(x.type(dtype))
    y_var = Variable(y.type(dtype).long())

    scores = model(x_var)
    loss = loss_fn(scores, y_var)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Check accuracy of model
def check_accuracy(model, loader, dtype):
  model.eval()
  model.cuda()
  num_correct, num_samples = 0, 0
  i = 0
  for x, y in loader:
    x_var = Variable(x.type(dtype), volatile=True)
    scores = model(x_var)
    _, preds = scores.data.cpu().max(1)
    num_correct += (preds == y).sum()
    num_samples += x.size(0)
  acc = float(num_correct) / num_samples
  return acc

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
