import torch
import torch.nn as nn
import os, sys
from torch.optim import lr_scheduler
from collections import OrderedDict, namedtuple
from torchvision import  models




class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        kernel_size = 3
        model = []
        ndf = 32
        num = 4

        model += [nn.Conv2d(3, ndf, 7, stride=2),
                  nn.BatchNorm2d(affine=True, num_features=ndf),
                  nn.ReLU(True)]

        for i in range(num):
            mult = 2 ** i
            model += [nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size, stride=2),
                      nn.BatchNorm2d(affine=True, num_features=ndf * mult * 2),
                      nn.ReLU(True)]

        self.conv_model = nn.Sequential(*model)
        self.avg = nn.AdaptiveAvgPool2d([1,1])
        self.cls = nn.Conv2d(ndf * (2**num), 2, kernel_size=1, stride=1)

    def num_flat_features(self, x):
        num_features = 1
        for size in x.size()[1:]:
            num_features *= size

        return num_features

    def forward(self, input):
        # print(input)
        conv_out = self.conv_model(input)
        # self.conv_out = self.conv_out.view(-1, self.num_flat_features(self.conv_out))
        conv_out = self.avg(conv_out)
        out = self.cls(conv_out).squeeze()
        # print("out size", out.size())

        return out

class Resnet(nn.Module):
    def __init__(self, num, opt):
        super(Resnet, self).__init__()
        self.num_class = opt.num_class
        self.model = getattr(models, "resnet%d"%num)(True)
        self.set_parameter_requires_grad(self.model, True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_class)

    def parameters(self):
        paramters_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                paramters_to_update.append(param)
                print("\t", name)
        return paramters_to_update


    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, input):
        return self.model(input)




class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.model = None
        self.opt = None

    def setup(self):
        self.print_networks(verbose=True)

    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in self.model.parameters():
            num_params += param.numel()
        if verbose:
            print(self.model)
        print('Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')


    def update_lr_scheduler(self, start_epoch):
        self.lr_scheduler = self.get_scheduler(self.optim, start_epoch, self.opt)


    def get_scheduler(self, optimizer, start_epoch, opt):
        if opt.lr_policy == 'lambda':
            def lambda_rule(epoch):
                lr_l = max(0.0, 1.0 - max(0, epoch - opt.niter + start_epoch) / float(opt.niter_decay + 1))
                return lr_l

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

        elif opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)

        elif opt.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01,
                                                       patience=5)
        elif opt.lr_policy == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)

        return scheduler

    def update_lr(self):
        self.lr_scheduler.step()


    def save_model(self, epoch, name):
        save_path = os.path.join(self.checkpoints_dir, "net_epoch_{}_{}.pth".format(epoch, name))
        # self.model.state_dict()

        states = {"epoch": epoch + 1,
                  "optimizer": self.optim.state_dict()}

        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            states["state_dict"] = self.model.state_dict()

            torch.save(states, save_path)
            # self.model.cuda(self.gpu_ids[0])
            # print(self.model.state_dict())

        else:
            states["state_dict"] = self.model.state_dict()

            torch.save(states, save_path)


class classifier(BaseModel):
    def initialize(self, opt):
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device("cuda:{}".format(self.gpu_ids[0])) if len(self.gpu_ids) > 0 else torch.device("cpu")
        self.checkpoints_dir = opt.checkpoints_dir
        self.lr_policy = opt.lr_policy
        self.opt = opt

        # model = SimpleNet()
        model = Resnet(18, opt)
        model = model.to(self.device)

        self.model = model
        self.optim = torch.optim.Adam(self.model.parameters(), lr=opt.lr, weight_decay=1e-5)
        self.crition = nn.CrossEntropyLoss()


    def set_input(self, data):
        self.img, self.label = data["img"], data["label"]

        self.img = self.img.to(self.device)
        self.label = self.label.to(self.device)


    def get_current_states(self):
        States = namedtuple("States", ["images", "scalars"])

        states = States(images={}, scalars={})

        states.scalars["loss"] = self.calc_loss()
        states.scalars["acc"] = self.calc_acc()
        states.scalars["lr"] = self.optim.param_groups[0]["lr"]
        states.images["img"] = self.img

        return states


    def forward(self):
        self.logtis = self.model(self.img)


    def calc_loss(self):
        loss = self.crition(self.logtis, self.label)
        return loss

    def calc_acc(self):
        preds = torch.argmax(self.logtis, dim=1)

        acc = torch.mean((preds == self.label).double())

        return acc


    def optimize_parameters(self):

        self.forward()
        self.loss = self.calc_loss()
        self.acc = self.calc_acc()

        self.optim.zero_grad()

        self.loss.backward()

        self.optim.step()




