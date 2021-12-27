from torchvision.models import resnet50
import torch.nn as nn
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from utils.experiman import manager
from torch.utils.data import DataLoader
import torch.optim as optim
from Trainer import StandardTrainer

def get_resnet50_model(opt):
    if opt.pretrained:
        model = resnet50(pretrained=True)
    else:
        model = resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, opt.class_num)
    return model

def get_cifar10_dataloader(opt):
    # Standard Transform to CIFAR10
    transforms_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465)
                             ,(0.2023, 0.1994, 0.2010))
    ])
    transforms_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465)
                             ,(0.2023, 0.1994, 0.2010))
    ])

    aug_trainset = CIFAR10(root=opt.data_dir, transform=transforms_train)
    raw_trainset = CIFAR10(root=opt.data_dir, transform=transforms_test)
    testset = CIFAR10(root=opt.data_dir, train=False, transform=transforms_test)

    aug_trainloader = DataLoader(
        aug_trainset, batch_size=opt.batch, shuffle=True,
        drop_last=True, num_workers=opt.num_workers)
    raw_trainloader = DataLoader(
        raw_trainset, batch_size=opt.batch, shuffle=False,
        drop_last=False, num_workers=opt.num_workers)
    testloader = DataLoader(
        testset, batch_size=opt.batch, shuffle=False,
        drop_last=False, num_workers=opt.num_workers)

    return aug_trainloader, raw_trainloader, testloader

def add_parser_argument(parser):
    ## ======================== Data ==========================
    parser.add_argument('--batch', default=64, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    ## ======================== Model ==========================
    parser.add_argument('--class_num', type=int, default=10)
    ## ===================== Training =========================
    '''Already set Most of the parameter by default'''
    parser.add_argument('--pretrained', action='store_true')
    ## ==================== Optimization ======================
    parser.add_argument('--epoch', default=15, type=int)
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--lr_schedule', type=str, default='cos')

def main():
    parser = manager.get_basic_arg_parser()
    add_parser_argument(parser)
    opt = parser.parse_args()
    manager.setup(opt, third_party_tools=('tensorboard',))
    logger = manager.get_logger()
    device = 'cuda'

    # Data
    logger.info('==> Preparing data..')
    trainloader, raw_trainloader, testloader = get_cifar10_dataloader(opt)

    # Model
    logger.info('==> Building models..')
    model = get_resnet50_model(opt).to(device)

    # Optimizer
    if opt.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay,
            nesterov=False)
    elif opt.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, betas=(0.9, 0.999))
    else:
        raise NotImplementedError

    if opt.lr_schedule == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.epoch)
    else:
        raise NotImplementedError

    # Criterion
    criterion = nn.CrossEntropyLoss()
    # Train
    trainer = StandardTrainer(manager=manager,
                  model=model,
                  dataloaders=(trainloader, raw_trainloader, testloader),
                  iters_per_epoch=(len(trainloader), len(testloader)),
                  criterion=criterion,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  num_epochs=opt.epoch,
                  device=device)
    trainer.train()

if __name__ == "__main__":
    main()
