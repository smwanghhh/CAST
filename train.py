import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import os, torch
import argparse
import Networks
from dataset import RafDataSet, FER
import torch.nn.functional as F
import math
from thop import profile
import image_utils as util
import random
from einops import rearrange, repeat
from torchvision.utils import save_image
from randaugment import  RandAugmentMC
from sklearn.cluster import KMeans
global epoch
seed = 1314 
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



def _init_fn(worker_id):
    np.random.seed(int(seed))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data1', type=str, default='rafdb', help='source data')
    parser.add_argument('--data2', type=str, default='fer', help='target data.')
    parser.add_argument('--idx', type=int, default=3, help='10-folder cross validation')
    parser.add_argument('-c', '--checkpoint', type=str, default= None,
                        help='load model')  
    parser.add_argument('--backbone', type=str, default='resnet18', help='Backobone, resnet18, resnet50 or mobilenet_v2.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=10, type=int, help='Number of data loading workers (default: 10)')
    parser.add_argument('--epochs', type=int, default=30, help='Total training epochs.')
    parser.add_argument('--w1', type=float, default=4, help='classification loss weight')
    parser.add_argument('--w2', type=float, default=0.3, help='affinity loss weight')
    parser.add_argument('--w3', type=float, default=0.1, help='weight loss weight')
    parser.add_argument('--phi', type=float, default=1.4, help='weight loss weight')
    return parser.parse_args()



def run_training():
    args = parse_args()
    model_path = os.path.join('./models', args.data1 + '_' + args.data2)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    figsize = [10, 10]

    print('---------------------------------------------------------------------------------------')
    print('Training %s with source data %s and target data %s, idx %s:'%(args.backbone, args.data1, args.data2, args.idx))
    print('w1:%s        w2:%s      w3:%s'%(str(args.w1), str(args.w2), str(args.w3)))
    print('---------------------------------------------------------------------------------------')
     
    if args.backbone == 'resnet18':
        train_batch = 128
        test_batch = 128
        pre_epochs = 30
        
    elif args.backbone == 'resnet50':
        train_batch = 50
        test_batch = 100
        pre_epochs = 30
        
    elif args.backbone == 'mobilenet_v2':
        train_batch = 64
        test_batch = 128
        pre_epochs = 30

    data_transforms = {
        "train": transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(20),
                                transforms.RandomCrop(224, padding=32)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25))]),

        "test": transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),

        'augment': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(20),
                                transforms.RandomCrop(224, padding=32)], p=0.5),
        RandAugmentMC(n=2, m=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25))]),

    }

    ###source data
    if args.data1 == 'rafdb':
        train_dataset = RafDataSet('../datasets/raf-basic/', phase='train', transform=data_transforms['train'], strong_transform = data_transforms['augment'],
                                   basic_aug=False)
        val_dataset = RafDataSet('../datasets/raf-basic/', phase='test', transform=data_transforms['test'], strong_transform = None)
        class_num = 7
        class_name = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']
        source_train = train_dataset
        source_test = val_dataset
        source_val_num = val_dataset.__len__()

    else:
        raise ValueError('Please input right source data')

    if args.data2 == 'fer':
        train_dataset = FER('../datasets/fer2013/', phase='train', transform=data_transforms['train'], strong_transform = data_transforms['augment'],
                            basic_aug=False)
        val_dataset = FER('../datasets/fer2013/', phase='test', transform=data_transforms['test'], strong_transform = None)
        class_num = 7
        class_name = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']
        target_train = train_dataset
        target_test = val_dataset
        target_val_num = val_dataset.__len__()
        
    else:
        raise ValueError('Please input right target data')


    train_loader_source = torch.utils.data.DataLoader(source_train,
                                                      batch_size=train_batch,
                                                      num_workers=args.workers,
                                                      shuffle=True,
                                                      pin_memory=True)


    val_loader_source = torch.utils.data.DataLoader(source_test,
                                                    batch_size=test_batch,
                                                    num_workers=args.workers,
                                                    shuffle=False,
                                                    pin_memory=True)

    train_loader_target = torch.utils.data.DataLoader(target_train,
                                                      batch_size=train_batch,
                                                      num_workers=args.workers,
                                                      shuffle=True,
                                                      pin_memory=True)

    val_loader_target = torch.utils.data.DataLoader(target_test,
                                                    batch_size=test_batch,
                                                    num_workers=args.workers,
                                                    shuffle=False,
                                                    pin_memory=True)


    model = Networks.Model(backbone=args.backbone, num_classes=class_num)

    if args.checkpoint:
        print("Loading pretrained weights...", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'], strict=True)

    param = model.parameters()#list(model.feature.parameters()) + list(model.fc.parameters())
    optimizer = torch.optim.Adam(param, args.lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss(reduce = False)
    
    ####training the model on the source dataset
    best_acc = 0.
    for i in range(0, pre_epochs):
        train_loss1, train_loss2, train_loss3 = 0.0, 0.0, 0.0
        count = 0
        model.train()
        bingo_cnt = 0.
        for batch_i, (imgs, _, targets) in enumerate(train_loader_source):
            imgs = imgs.cuda()
            targets = targets.cuda()
            output = model(imgs, targets, None, 'train', 'source')
            cls_loss = torch.mean(criterion(output[0], targets))               
            ###classifiers modulation
            fc_weight = model.fc.weight
            fc_weight_norm = torch.norm(fc_weight, dim = 1).unsqueeze(1)
            fc_weight_ = fc_weight.mm(torch.transpose(fc_weight, 1, 0))
            fc_weight_norm_ = fc_weight_norm.mm(torch.transpose(fc_weight_norm, 1, 0))
            weight_loss = torch.mean(((fc_weight_ / fc_weight_norm_ -  torch.eye(fc_weight.shape[0]).cuda()) + 1) / 2)

            aff_loss = output[1]
            loss = cls_loss * args.w1 #+ aff_loss * args.w2 + weight_loss * args.w3    
            train_loss2 += aff_loss

            loss.backward()
            optimizer.step()

            _, predicts = torch.max(output[0], 1)
            correct_or_not = torch.eq(predicts, targets)
            bingo_cnt += correct_or_not.sum().cpu().numpy()
            train_loss1 += cls_loss
            train_loss3 += weight_loss
            optimizer.zero_grad()
        count += 1
        train_acc = bingo_cnt / (count * train_batch)
        train_acc = np.around(train_acc, 4)
        train_loss1 = train_loss1 / count
        train_loss2 = train_loss2 / count
        train_loss3 = train_loss3 / count
        
        print('[Epoch %d] Training accuracy: %.4f.   Classification Loss: %.3f   Affinity Loss: %.3f  Weight Loss: %.3f  LR: %.6f' %
                  (i, train_acc, train_loss1, train_loss2, train_loss3, optimizer.param_groups[0]["lr"]))
        scheduler.step()
        best_acc = test(model, optimizer, val_loader_target, criterion, target_val_num, best_acc, model_path, i, args)
    
    checkpoint = torch.load(os.path.join(model_path, args.backbone + '_' + args.data1 + '_' + args.data2 + '_' + str(best_acc) + ".pth"))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    optimizer.param_groups[0]["lr"] = args.lr  * 0.09

    for i in range(0, args.epochs):
        train_loss1, train_loss2 = 0, 0
        count = 0
        confident_num = 0

        for batch_i, (imgs, imgs_aug, gt_target) in enumerate(train_loader_target):
            try:
                source_imgs, _, source_targets = next(source_train_iter)
            except:
                source_train_iter = iter(train_loader_source)
                source_imgs, _, source_targets = next(source_train_iter)
            model.eval()
            ####forward the model get pseudo labels
            with torch.no_grad():
                out, fea = model(imgs.cuda(), None, None, 'test', 'target')

            targets, con_idx, threshold, label_dis = annotate_target(out, class_num, i, args.epochs, args.phi)

            model.train()
            source_con_idx = torch.ones(source_imgs.shape[0])
            ###cmbine the source and target batch
            train_imgs = torch.cat((source_imgs, imgs_aug), 0).cuda()
            train_targets = torch.cat((source_targets, targets), 0).cuda()
            train_con_idx = torch.cat((source_con_idx, con_idx), 0).cuda()

            output = model(train_imgs, train_targets, train_con_idx, 'train', 'target')
            ##classification _loss
            cls_loss = torch.mean(criterion(output[0], train_targets) * train_con_idx)

            fc_weight = model.fc.weight
            fc_weight_norm = torch.norm(fc_weight, dim = 1).unsqueeze(1)
            fc_weight_ = fc_weight.mm(torch.transpose(fc_weight, 1, 0))
            fc_weight_norm_ = fc_weight_norm.mm(torch.transpose(fc_weight_norm, 1, 0))
            weight_loss = torch.mean(((fc_weight_ / fc_weight_norm_ -  torch.eye(fc_weight.shape[0]).cuda()) + 1) / 2)

            aff_loss = output[1]
            loss = cls_loss * args.w1 + aff_loss * args.w2 + weight_loss * args.w3 
            train_loss2 += aff_loss
            loss.backward()
            optimizer.step()
            train_loss1 += cls_loss
            optimizer.zero_grad()
            try:
                confident_num += (con_idx != 0).nonzero().squeeze().shape[0]
            except:
                confident_num = 0
        count += 1       
        scheduler.step()
        train_loss1 = train_loss1 / count
        train_loss2 = train_loss2 / count

        print('[Epoch %d]  Confident_Num: %d    Classification Loss: %.3f   Affinity Loss: %.3f   LR: %.6f' %
                  (i, confident_num, train_loss1, train_loss2, optimizer.param_groups[0]["lr"]))
        best_acc = test(model, optimizer, val_loader_target, criterion, target_val_num, best_acc, model_path, i, args)
    print("best_acc %s " % str(best_acc))
    


def test(model, optimizer, val_loader_target, criterion, num, best_acc, path, epoch, args):

    with torch.no_grad():
        val_loss = 0.0
        iter_cnt = 0
        bingo_cnt = 0
        preds, labels = [], []
        model.eval()
        for batch_i, (imgs, targets) in enumerate(val_loader_target):
            out, _ = model(imgs.cuda(), targets, None, mode = 'test')
            targets = targets.cuda()
            loss = torch.mean(criterion(out, targets))
            val_loss += loss
            iter_cnt += 1
            _, predicts = torch.max(out, 1)
            correct_or_not = torch.eq(predicts, targets)
            bingo_cnt += correct_or_not.sum().cpu()
            preds.append(predicts.cpu())
            labels.append(targets.cpu())

        val_loss = val_loss / iter_cnt
        val_acc = bingo_cnt.float() / float(num)
        val_acc = np.around(val_acc.numpy(), 4)
        print("[Epoch %d] Target Validation accuracy:%.4f.  Loss:%.3f" % (epoch, val_acc, val_loss))
        class_acc, _ = util.make_confucion_matrix(preds, labels)
        pred = torch.cat(preds, dim=0)
        labelss = torch.cat(labels, dim=0)
        mean_acc = val_acc #np.mean(class_acc)
        
        if mean_acc > best_acc:
            try:
                os.remove(os.path.join(path, args.backbone + '_' + args.data1 + '_' + args.data2 + '_' + str(best_acc) + ".pth"))
            except:
                pass
            best_acc = mean_acc
            save_data = {'model': model.state_dict(),
                         'optimizer': optimizer.state_dict()}
            torch.save(save_data,
                       os.path.join(path, args.backbone + '_' + args.data1 + '_' + args.data2 + '_' + str(best_acc) + ".pth"))
                       
            print("best_acc %s " % str(best_acc))

    return best_acc

def annotate_target(pred, class_num, i, sum, phi):

    pred = pred.cpu()
    pred = F.softmax(pred, 1)
    pred_values, pred_targets = torch.max(pred, dim= 1) ###select max predictions
    ###calculate the mean predictions
    max_index = F.one_hot(pred_targets, class_num)
    preds_mean = np.transpose((pred * max_index).detach().numpy())
    class_sum = [np.sum(pred_mean) for pred_mean in preds_mean]
    class_idx = [len(np.where(pred_mean > 0)[0]) for pred_mean in preds_mean]
    class_mean = np.array([class_sum[index] / class_idx[index] if class_idx[index]!=0 else 0 for index in range(len(class_idx))])
    class_mean = np.array([mean * phi * (sum/(sum-i)) for mean in class_mean])  
    class_mean = torch.from_numpy(np.minimum(class_mean, 0.9))
    threshold = class_mean.numpy()
    batch_mean = torch.index_select(class_mean, 0, pred_targets).detach().numpy()
    confident_idx = torch.from_numpy((pred_values.detach().numpy() > batch_mean).nonzero()[0]) 

    ###  show condident sample class distribution
    t = pred_targets.index_select(0, confident_idx).numpy()
    label_dis = [np.sum(t == 0), np.sum(t == 1), np.sum(t == 2),
                      np.sum(t == 3), \
                      np.sum(t == 4), np.sum(t == 5), np.sum(t == 6)]
                      
    # print('Confident dataset distribute: %d, %d, %d, %d, %d, %d, %d' % (
    #     label_dis[0], label_dis[1], label_dis[2], label_dis[3], \
    #     label_dis[4], label_dis[5], label_dis[6]))
    ones = torch.ones(confident_idx.shape[0]) 
    confident_idx = torch.zeros(pred.shape[0]).index_put([torch.LongTensor(confident_idx)], ones)
    return pred_targets, confident_idx, threshold, label_dis
    

if __name__ == "__main__":
    class RecorderMeter():
        pass

    acc = run_training()
