import torch
import torchvision
from torchvision.datasets import ImageFolder
import argparse
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torchvision.transforms as T
from torch.autograd import Variable
'''
to run: python test_model.py --test_dir ~/data/test --use_gpu
'''
parser = argparse.ArgumentParser()
parser.add_argument('--test_dir', default='data/test')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def main(folder):
    #dtype = torch.FloatTensor
    #if args.use_gpu:
    dtype = torch.cuda.FloatTensor

    val_transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ])
    test_dset = ImageFolder(folder, transform=val_transform)
    test_loader = DataLoader(test_dset, batch_size=32, num_workers=4)

    model = load_model('classifier.pth',test_dset,dtype)
    test_acc = check_accuracy(model, test_loader, dtype)

# Load previously trained model from checkpoint
def load_model(checkpoint_path,test_dset,dtype):
    chpt = torch.load(checkpoint_path)

    model =torchvision.models.resnet18(pretrained=True)

    model.class_to_idx = chpt['class_to_idx']

    num_classes = len(test_dset.classes)
    print(test_dset.classes)
    model.fc = nn.Linear(model.fc.in_features, 5)

    model.type(dtype)
    loss_fn = nn.CrossEntropyLoss().type(dtype)

    for param in model.parameters():
       param.requires_grad = False
    for param in model.fc.parameters():
       param.requires_grad = True

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    # Put the classifier on the pretrained network
   # model.classifier = classifier

    model.load_state_dict(chpt['state_dict'])

    return model

def check_accuracy(model, loader, dtype):
  model.eval()
  model.cuda()
  num_correct, num_samples = 0, 0

  i = 0

  for x, y in loader:
    x_var = Variable(x.type(dtype))
    scores = model(x_var)
    _, preds = scores.data.cpu().max(1)


    num_correct += (preds == y).sum()
    num_samples += x.size(0)

  acc = float(num_correct) / num_samples*100

  return acc

if __name__ == '__main__':
  args = parser.parse_args()
  main(args.test_dir)
