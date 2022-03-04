import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader,Dataset
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
class AlexNet(nn.Module):
    def __init__(self,):
        super( AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 96,stride = 2, padding = 2, padding_mode= 'zeros',kernel_size=(11,5))
        self.mp1 = nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 96, out_channels = 256,kernel_size = 5, stride = 1, padding_mode = 'zeros')
        self.mp2 = nn.MaxPool2d(kernel_size = 3,stride = 2,padding = 1 )
        self.conv3 = nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3, stride = 1, padding= 1,padding_mode = 'zeros')
        self.conv4 = nn.Conv2d( in_channels=384, out_channels=384, kernel_size = 3, stride = 1, padding=1, padding_mode='zeros' )
        self.conv5 = nn.Conv2d(in_channels = 384, out_channels=256,kernel_size=(4,3),stride = 1, padding = 1, padding_mode = 'zeros')
        self.mp3 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.fc1 = nn.Linear( 256*46*13, 256 )
        self.fc2 = nn.Linear( 256, 1280 )
        self.fc3 = nn.Linear( 1280, 276 )
    def forward( self, x ):
        '''
        :param x: (n_samples, n_channels,time stamp, subcarrier)
        '''
        x = F.relu(self.conv1(x))
        x = self.mp1(x)
        x = F.relu( self.conv2(x))
        x = self.mp2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view( -1, 256*46*13 )
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.normalize( x, dim=-1, p=2 )
        x = F.relu(self.fc3(x))
        return x
alex = AlexNet()
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

num_ftrs = alex.fc1.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)
model_conv = model_conv.to(device)
criterion = nn.CrossEntropyLoss()
# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)
# # Device configuration
# device = torch.device( 'cuda' if torch.cuda.is_available( ) else 'cpu' )
#
# # Hyper-parameters
# num_epochs = 5
# batch_size = 4
# learning_rate = 0.001
#
# # dataset has PILImage images of range [0, 1].
# # We transform them to Tensors of normalized range [-1, 1]
# transform = transforms.Compose(
#         [ transforms.ToTensor( ),
#           transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) ) ] )
# # CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
# train_dataset = torchvision.datasets.CIFAR10( root='./data', train=True,
#                                               download=True, transform=transform )
# test_dataset = torchvision.datasets.CIFAR10( root='./data', train=False,
#                                              download=True, transform=transform )
# train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=batch_size,
#                                             shuffle=True )
# test_loader = torch.utils.data.DataLoader( test_dataset, batch_size=batch_size,
#                                            shuffle=False )
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# class ConvNet( nn.Module ):
#     def __init__( self ):
#         super( ConvNet, self ).__init__( )
#         print('In the convnet init')
#         self.conv1 = nn.Conv2d( 3, 12, 5 )
#         self.pool = nn.MaxPool2d( 2, 2 )
#         self.conv2 = nn.Conv2d( 12, 16, 5 )
#         self.fc1 = nn.Linear( 16 * 5 * 5, 120 )
#         self.fc2 = nn.Linear( 120, 84 )
#         self.fc3 = nn.Linear( 84, 10 )
#     def forward( self, x ):
#         # -> n, 3, 32, 32
#         x = self.pool( F.relu( self.conv1( x ) ) )  # -> n, 6, 14, 14
#         x = self.pool( F.relu( self.conv2( x ) ) )  # -> n, 16, 5, 5
#         x = x.view( -1, 16 * 5 * 5 )  # -> n, 400
#         x = F.relu( self.fc1( x ) )  # -> n, 120
#         x = F.relu( self.fc2( x ) )  # -> n, 84
#         x = self.fc3( x )  # -> n, 10
#         return x
# model = ConvNet( ).to( device )
# criterion = nn.CrossEntropyLoss( )
# optimizer = torch.optim.SGD( model.parameters( ), lr=learning_rate )
#
# n_total_steps = len( train_loader )
# for epoch in range( num_epochs ):
#     for i, (images, labels) in enumerate( train_loader ):
#         # origin shape: [4, 3, 32, 32] = 4, 3, 1024
#         # input_layer: 3 input channels, 6 output channels, 5 kernel size
#         images = images.to( device )
#         labels = labels.to( device )
#
#         # Forward pass
#         outputs = model( images )
#         loss = criterion( outputs, labels )
#
#         # Backward and optimize
#         optimizer.zero_grad( )
#         loss.backward( )
#         optimizer.step( )
#
#         if (i + 1) % 2000 == 0:
#             print( f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item( ):.4f}' )
#
# print( 'Finished Training' )
# PATH = './cnn.pth'
# torch.save( model.state_dict( ), PATH )
#
# with torch.no_grad( ):
#     n_correct = 0
#     n_samples = 0
#     n_class_correct = [ 0 for i in range( 10 ) ]
#     n_class_samples = [ 0 for i in range( 10 ) ]
#     for images, labels in test_loader:
#         images = images.to( device )
#         labels = labels.to( device )
#         outputs = model( images )
#         # max returns (value ,index)
#         _, predicted = torch.max( outputs, 1 )
#         n_samples += labels.size( 0 )
#         n_correct += (predicted == labels).sum( ).item( )
#
#         for i in range( batch_size ):
#             label = labels[ i ]
#             pred = predicted[ i ]
#             if (label == pred):
#                 n_class_correct[ label ] += 1
#             n_class_samples[ label ] += 1
#
#     acc = 100.0 * n_correct / n_samples
#     print( f'Accuracy of the network: {acc} %' )
#
#     for i in range( 10 ):
#         acc = 100.0 * n_class_correct[ i ] / n_class_samples[ i ]
#         print( f'Accuracy of {classes[ i ]}: {acc} %' )