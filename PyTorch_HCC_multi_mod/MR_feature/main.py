import os
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from opts import parse_opts
from model import generate_model
from dataset import TrainSet, ValidSet
from utils import Logger, OsJoin
from train import train_epoch
from validation import val_epoch
from tensorboardX import SummaryWriter
from focal_loss import FocalLoss


def run(fold_id, opt):
    use_focal_loss = False
    if opt.root_path != '':
        result_path = OsJoin(opt.root_path, opt.result_path)
        event_path = OsJoin(opt.root_path, opt.event_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
    opt.arch ='{}-{}'.format(opt.model_name,opt.model_depth)
    #print(opt)

    print('-'*50, 'RUN FOLD %s' % str(fold_id), '-'*50)

    model, parameters = generate_model(opt)
    # print(model)
    if use_focal_loss:
        criterion = FocalLoss(gamma=2, alpha=0.75)
    else:
        criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if not opt.no_train:
        training_data = TrainSet(fold_id=fold_id)
        # print('training data', training_data)
        train_loader = DataLoader(training_data, batch_size=opt.batch_size, shuffle=True,
                                  num_workers=opt.n_threads, pin_memory=True)
        if opt.pretrain_path:
            log_path = OsJoin(result_path, opt.data_type, opt.model_name + '_' + str(opt.model_depth) + '_pretrain',
                              'logs_fold%s' % str(fold_id))
            event_path = OsJoin(event_path, opt.data_type, opt.model_name + '_' + str(opt.model_depth) + '_pretrain',
                                'logs_fold%s' % str(fold_id))
        elif not opt.pretrain_path:
            log_path = OsJoin(result_path, opt.data_type, opt.model_name + '_' + str(opt.model_depth),
                              'logs_fold%s' % str(fold_id))
            event_path = OsJoin(event_path, opt.data_type, opt.model_name + '_' + str(opt.model_depth),
                                'logs_fold%s' % str(fold_id))

        if not os.path.exists(log_path):
            os.makedirs(log_path)
        train_logger = Logger(
            OsJoin(log_path, 'train.log'),
            ['epoch', 'loss', 'acc', 'lr'])

        train_batch_logger = Logger(
            OsJoin(log_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])


        if opt.train_pretrain is not '':
            params = [
                {'params': filter(lambda p: p.requires_grad, parameters['base_parameters']), 'lr': opt.learning_rate*0.001},
                {'params': filter(lambda p: p.requires_grad, parameters['new_parameters']), 'lr': opt.learning_rate}
            ]
        else:
            params = [{'params': filter(lambda p: p.requires_grad, parameters), 'lr': opt.learning_rate}]
        print('params', params)
        # optimizer = optim.Adam(params, weight_decay=opt.weight_decay)
        optimizer = optim.SGD(parameters, lr=opt.learning_rate, momentum=0.9, weight_decay=opt.weight_decay)
        print('opt.learning_rate', opt.learning_rate)

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min',
                                                   factor=opt.lr_decay_factor, patience=opt.lr_patience,
                                                   min_lr=1e-4)
    if not opt.no_val:
        validation_data = ValidSet(fold_id=fold_id)
        # print('validation_data', validation_data)
        val_loader = DataLoader(validation_data, batch_size=opt.batch_size, shuffle=False,
                                                    num_workers=opt.n_threads, pin_memory=True)
        val_logger = Logger(OsJoin(log_path, 'val.log'), ['epoch', 'loss', 'acc', 'recall'])

    # if opt.pretrain_path:
    #     print('loading checkpoint{}'.format(opt.pretrain_path))
    #     checkpoint = torch.load(opt.pretrain_path)
    #     assert opt.arch==checkpoint['arch']
    #
    #     opt.begin_epoch = checkpoint['epoch']
    #     model.load_state_dict(checkpoint['state_dict'])
    #     if not opt.no_train:
    #         optimizer.load_state_dict(checkpoint['optimizer'])

    #print('run')
    writer = SummaryWriter(logdir=event_path)
    for i in range(opt.begin_epoch, opt.n_epochs+1):
        if not opt.no_train:
            # print('criterion', criterion)
            model = train_epoch(i, fold_id, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger, writer)
        if not opt.no_val:
            validation_loss = val_epoch(i, val_loader, model, criterion, opt, val_logger, writer)
        if not opt.no_train and not opt.no_val:
            scheduler.step(validation_loss)
            lr = optimizer.param_groups[0]["lr"]
            print('lr', lr)
            writer.add_scalar('lr', lr, i)
    writer.close()
    print('-'*47, 'FOLD %s FINISHED'%str(fold_id), '-'*48)
    print("===>save models...")
    os.chdir('/home/amax/Wendy/nnUNet/ouput_kaggle/results')
    print(os.getcwd())
    # print('model', model)
    torch.save(model, 'net_model_fold{}.pkl'.format(fold_id))
# if opt.test:
#     test_data = TestSet()
#     test_loader = torch.utils.data.DataLoader(test_data, batch_size = opt.batch_size, shuffle=False,
#                                                         num_workers = opt.n_threads, pin_memory=True)
#     test.test(test_loader, model, opt, test_data.label)


if __name__ == '__main__':
    opt = parse_opts()
    opt.batch_size = 64
    opt.n_epochs = 1000
    opt.no_val = False
    opt.n_threads = 4
    # opt.
    print('opt.n_threads', opt.n_threads)
    print('opt.no_val', opt.no_val)
    print(opt.model_name)
    print(opt.model_depth)
    run('3', opt)
    # ????????????
    # for fold_id in range(1, opt.n_fold + 1):
    #     run(fold_id, opt)

