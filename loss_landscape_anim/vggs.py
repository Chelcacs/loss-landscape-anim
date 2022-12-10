'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from loss_landscape_anim.model import GenericModel
import torch.nn.functional as F


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(GenericModel):
    def __init__(self, vgg_name,lerning_rate,optimizer='adam', custom_optimizer=None, gpus=0):
        super(VGG, self).__init__(optimizer,lerning_rate,custom_optimizer, gpus)
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    def loss_fn(self, y_pred, y):
        """Loss function."""
        return F.cross_entropy(y_pred, y)
    
    def training_step(self, batch, batch_idx):
        """Training step for a batch of data.

        The model computes the loss and save it along with the flattened model params.
        """
        X, y = batch
        y_pred = self(X)
        # Get model weights flattened here to append to optim_path later
        flat_w = self.get_flat_params()
        loss = self.loss_fn(y_pred, y)

        preds = y_pred.max(dim=1)[1]  # class
        accuracy = self.accuracy(preds, y)

        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss, "accuracy": accuracy, "flat_w": flat_w}

    def training_epoch_end(self, training_step_outputs):
        """Only save the last step in each epoch.

        Args:
            training_step_outputs: all the steps in this epoch.
        """
        self.optim_path.extend(training_step_outputs)


def VGG11():
    return VGG('VGG11')

def VGG13():
    return VGG('VGG13')

def VGG16():
    return VGG('VGG16')

def VGG19():
    return VGG('VGG19')

def test():
    net = VGG('VGG11')
    x = torch.randn(2,1,45,45)
    print(net(Variable(x)).size())

if __name__ == '__main__':
    test()