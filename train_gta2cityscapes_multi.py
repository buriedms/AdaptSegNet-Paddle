import argparse
import numpy as np
import pickle
import scipy.misc

import paddle
import paddle.nn as nn
from paddle.io import DataLoader
import paddle.optimizer as optim
import paddle.nn.functional as F
from paddle.optimizer.lr import PolynomialDecay

import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import random

from model.deeplab_multi import DeeplabMulti
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2D
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 0 # todo 4 -> 0
DATA_DIRECTORY = './data/GTA5'
DATA_LIST_PATH = './dataset/gta5_list/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '1280,720'
DATA_DIRECTORY_TARGET = './data/Cityscapes/data'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
INPUT_SIZE_TARGET = '1024,512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 250000
NUM_STEPS_STOP = 150000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'model/DeepLab_resnet_pretrained_init-f81d91e8.pdparams'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
CHECKPOINT = './checkpoint/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001
GAN = 'Vanilla'

TARGET = 'cityscapes'
SET = 'train'

def gen_logger(save_path=None,name=None,chlr=False,mode='w'):
    import logging
    import os
    name='' if not name else name
    save_path='' if not save_path else save_path
    file_path=os.path.join(save_path,name)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO ,
        filename=file_path,
        filemode=mode
    )
    logger=logging.getLogger()
    if chlr and file_path:
        chlr = logging.StreamHandler()  # 输出到控制台的handler
        logger.addHandler(chlr)
    return logger

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT,
                        help="Where to save checkpoint of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--gan", type=str, default=GAN,
                        help="choose the GAN objective.")
    return parser.parse_args()


args = get_arguments()


def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    criterion = CrossEntropy2D()
    label=paddle.to_tensor(label,dtype=paddle.int64)
    return criterion(pred, label)

def main():
    """Create the model and start the training."""

    logger=gen_logger(save_path=args.checkpoint_dir,name=f'{args.model}_train.log')

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    gpu = args.gpu

    # Create network
    model = DeeplabMulti(num_classes=args.num_classes)
    saved_state_dict = paddle.load(args.restore_from)
    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        # Scale.layer5.conv2d_list.3.weight
        i_parts = i.split('.')
        # print i_parts
        if not args.num_classes == 19 or not i_parts[1] == 'layer5':
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            # print i_parts
    model.set_state_dict(new_params)

    model.train()

    # init D
    model_D1 = FCDiscriminator(num_classes=args.num_classes)
    model_D2 = FCDiscriminator(num_classes=args.num_classes)

    model_D1.train()

    model_D2.train()

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    trainloader = DataLoader(
        GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.batch_size,
                    crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    trainloader_iter = iter(trainloader)

    targetloader = DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                max_iters=args.num_steps * args.batch_size,
                                                crop_size=input_size_target,
                                                scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
                                                set=args.set),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    targetloader_iter = iter(targetloader)

    # implement model.optim_parameters(args) to handle different models' learning_rate setting
    learning_rate=PolynomialDecay(args.learning_rate,decay_steps=args.num_steps,power=args.power)
    optimizer = optim.Momentum(parameters=model.parameters(), learning_rate=learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)

    learning_rate_D1 = PolynomialDecay(args.learning_rate_D, decay_steps=args.num_steps, power=args.power)
    optimizer_D1 = optim.Adam(parameters=model_D1.parameters(), learning_rate=learning_rate_D1, beta1=0.9,beta2=0.99)

    learning_rate_D2 = PolynomialDecay(args.learning_rate_D, decay_steps=args.num_steps, power=args.power)
    optimizer_D2 = optim.Adam(parameters=model_D2.parameters(), learning_rate=learning_rate_D2, beta1=0.9,beta2=0.99)

    optimizer.clear_grad()
    optimizer_D1.clear_grad()
    optimizer_D2.clear_grad()

    if args.gan == 'Vanilla':
        bce_loss = paddle.nn.BCEWithLogitsLoss()
    elif args.gan == 'LS':
        bce_loss = paddle.nn.MSELoss()

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1

    for i_iter in range(args.num_steps):

        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0

        loss_seg_value2 = 0
        loss_adv_target_value2 = 0
        loss_D_value2 = 0

        optimizer.clear_grad()
        optimizer_D1.clear_grad()
        optimizer_D2.clear_grad()

        # train G

        # don't accumulate grads in D
        for param in model_D1.parameters():
            param.stop_gradient = True

        for param in model_D2.parameters():
            param.stop_gradient = True

        # train with source

        batch = next(trainloader_iter)
        images, labels, _, _ = batch
        images = paddle.to_tensor(images,dtype=paddle.float32)

        pred1, pred2 = model(images)
        pred1 = interp(pred1)
        pred2 = interp(pred2)

        loss_seg1 = loss_calc(pred1, labels, args.gpu)
        loss_seg2 = loss_calc(pred2, labels, args.gpu)
        loss = loss_seg2 + args.lambda_seg * loss_seg1

        # proper normalization
        loss = loss
        loss.backward()

        loss_seg_value1 += loss_seg1.cpu().numpy()[0]
        loss_seg_value2 += loss_seg2.cpu().numpy()[0]
        # loss_seg_value1 += loss_seg1.cpu().numpy()
        # loss_seg_value2 += loss_seg2.cpu().numpy()

        # train with target

        batch = next(targetloader_iter)
        images, _, _ = batch
        images = paddle.to_tensor(images,dtype=paddle.float32)

        pred_target1, pred_target2 = model(images)
        pred_target1 = interp_target(pred_target1)
        pred_target2 = interp_target(pred_target2)

        D_out1 = model_D1(F.softmax(pred_target1, axis=1))
        D_out2 = model_D2(F.softmax(pred_target2, axis=1))

        D_out1_label = paddle.full(D_out1.shape, source_label, dtype=paddle.float32)
        D_out2_label = paddle.full(D_out2.shape, source_label, dtype=paddle.float32)

        loss_adv_target1 = bce_loss(D_out1, D_out1_label)
        loss_adv_target2 = bce_loss(D_out2, D_out2_label)

        loss = args.lambda_adv_target1 * loss_adv_target1 + args.lambda_adv_target2 * loss_adv_target2
        loss = loss
        loss.backward()
        loss_adv_target_value1 += loss_adv_target1.cpu().numpy()[0]
        loss_adv_target_value2 += loss_adv_target2.cpu().numpy()[0]
        # loss_adv_target_value1 += loss_adv_target1.cpu().numpy()
        # loss_adv_target_value2 += loss_adv_target2.cpu().numpy()

        # train D

        # bring back requires_grad
        for param in model_D1.parameters():
            param.stop_gradient = False

        for param in model_D2.parameters():
            param.stop_gradient = False

        # train with source
        pred1 = pred1.detach()
        pred2 = pred2.detach()

        D_out1 = model_D1(F.softmax(pred1, axis=1))
        D_out2 = model_D2(F.softmax(pred2, axis=1))

        D_out1_label = paddle.full(D_out1.shape, source_label, dtype=paddle.float32)
        D_out2_label = paddle.full(D_out2.shape, source_label, dtype=paddle.float32)

        loss_D1 = bce_loss(D_out1, D_out1_label)
        loss_D2 = bce_loss(D_out2, D_out2_label)

        loss_D1 = loss_D1 / 2.0
        loss_D2 = loss_D2 / 2.0

        loss_D1.backward()
        loss_D2.backward()

        loss_D_value1 += loss_D1.cpu().numpy()[0]
        loss_D_value2 += loss_D2.cpu().numpy()[0]
        # loss_D_value1 += loss_D1.cpu().numpy()
        # loss_D_value2 += loss_D2.cpu().numpy()

        # train with target
        pred_target1 = pred_target1.detach()
        pred_target2 = pred_target2.detach()

        D_out1 = model_D1(F.softmax(pred_target1, axis=1))
        D_out2 = model_D2(F.softmax(pred_target2, axis=1))

        D_out1_label = paddle.full(D_out1.shape, target_label, dtype=paddle.float32)
        D_out2_label = paddle.full(D_out2.shape, target_label, dtype=paddle.float32)

        loss_D1 = bce_loss(D_out1, D_out1_label)
        loss_D2 = bce_loss(D_out2, D_out2_label)

        loss_D1 = loss_D1 / 2.0
        loss_D2 = loss_D2 / 2.0

        loss_D1.backward()
        loss_D2.backward()

        loss_D_value1 += loss_D1.cpu().numpy()[0]
        loss_D_value2 += loss_D2.cpu().numpy()[0]
        # loss_D_value1 += loss_D1.cpu().numpy()
        # loss_D_value2 += loss_D2.cpu().numpy()

        optimizer.step()
        optimizer_D1.step()
        optimizer_D2.step()

        print('exp = {}'.format(args.checkpoint_dir))
        logger.info('exp = {}'.format(args.checkpoint_dir))
        print(
            'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} loss_seg2 = {3:.3f} loss_adv1 = {4:.4f}, loss_adv2 = {5:.4f} loss_D1 = {6:.4f} loss_D2 = {7:.4f}'.format(
                i_iter, args.num_steps, loss_seg_value1, loss_seg_value2, loss_adv_target_value1,
                loss_adv_target_value2, loss_D_value1, loss_D_value2))
        logger.info('iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} loss_seg2 = {3:.3f} loss_adv1 = {4:.4f}, loss_adv2 = {5:.4f} loss_D1 = {6:.4f} loss_D2 = {7:.4f}'.format(
                i_iter, args.num_steps, loss_seg_value1, loss_seg_value2, loss_adv_target_value1,
                loss_adv_target_value2, loss_D_value1, loss_D_value2))
        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            paddle.save(model.state_dict(), osp.join(args.checkpoint_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
            paddle.save(model_D1.state_dict(),
                        osp.join(args.checkpoint_dir, 'GTA5_' + str(args.num_steps_stop) + '_D1.pth'))
            paddle.save(model_D2.state_dict(),
                        osp.join(args.checkpoint_dir, 'GTA5_' + str(args.num_steps_stop) + '_D2.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking checkpoint ...')
            paddle.save(model.state_dict(), osp.join(args.checkpoint_dir, 'GTA5_' + str(i_iter) + '.pth'))
            paddle.save(model_D1.state_dict(), osp.join(args.checkpoint_dir, 'GTA5_' + str(i_iter) + '_D1.pth'))
            paddle.save(model_D2.state_dict(), osp.join(args.checkpoint_dir, 'GTA5_' + str(i_iter) + '_D2.pth'))


if __name__ == '__main__':
    main()
