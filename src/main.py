import torch
import torch.nn as nn
import numpy as np
import time
import argparse
import random
from torch.multiprocessing import Pool

from loss import PIXOR_Loss
from datagen import get_data_loader
from model import PIXOR
# from old_model import PIXOR
from utils import get_model_name, load_config, plot_bev, plot_label_map, plot_pr_curve, get_bev, scan_to_image, label_map_to_image
from postprocess import filter_pred, compute_matches, compute_ap, post_process
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from config import base_dir
import torchvision
import cv2 as cv


def build_model(device, train=True):
    '''
    Build the model, loss function, optimizer and scheduler

    Parameters:
        device (torch.device): device to run everything on
        train (bool): whether to provide an optimizer and scheduler

    Returns:
        nn.Module: network
        nn.Module: loss function
        torch.optim: optimizer (if train = True)
        torch.optim: scheduler (if train = True)
    '''
    config = load_config()[0]

    # net = PIXOR(config['geometry'], config['use_bn'])
    net = PIXOR()
    loss_fn = PIXOR_Loss()

    if config['mGPUs'] and torch.cuda.device_count() >= 1:
        print('Using Multi-GPU')
        net = nn.DataParallel(net)

    net = net.to(device)
    loss_fn = loss_fn.to(device)
    if not train:
        return net, loss_fn

    optimizer = torch.optim.SGD(
        net.parameters(), lr=config['learning_rate'],
        momentum=config['momentum'], weight_decay=config['weight_decay'])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config['lr_decay_at'], gamma=0.1)

    return net, loss_fn, optimizer, scheduler


def eval_batch(config, net, loss_fn, loader, device, eval_range='all'):
    net.eval()
    if config['mGPUs']:
        net.module.set_decode(True)
    else:
        net.set_decode(True)
    
    cls_loss = 0
    loc_loss = 0
    all_scores = []
    all_matches = []
    log_images = []
    gts = 0
    preds = 0
    t_fwd = 0
    t_nms = 0

    log_img_list = random.sample(range(len(loader.dataset)), 10)

    with torch.no_grad():
        for i, data in enumerate(loader):
            tic = time.time()
            input, label_map, image_id = data
            input = input.to(device)
            label_map = label_map.to(device)
            tac = time.time()
            predictions = net(input)
            t_fwd += time.time() - tac
            loss, cls, loc = loss_fn(predictions, label_map)
            cls_loss += cls
            loc_loss += loc 
            t_fwd += (time.time() - tic)
            
            toc = time.time()
            # Parallel post-processing
            predictions = list(torch.split(predictions.cpu(), 1, dim=0))
            batch_size = len(predictions)
            with Pool (processes=3) as pool:
                preds_filtered = pool.starmap(filter_pred, [(config, pred) for pred in predictions])
            t_nms += (time.time() - toc)
            args = []
            for j in range(batch_size):
                _, label_list = loader.dataset.get_label(image_id[j].item())
                corners, scores = preds_filtered[j]
                gts += len(label_list)
                preds += len(scores)
                all_scores.extend(list(scores))
                if image_id[j] in log_img_list:
                    input_np = input[j].cpu().permute(1, 2, 0).numpy()
                    pred_image = get_bev(input_np, corners)
                    log_images.append(pred_image)

                arg = (np.array(label_list), corners, scores)
                args.append(arg)

            # Parallel compute matchesi
            
            with Pool (processes=3) as pool:
                matches = pool.starmap(compute_matches, args)
            
            for j in range(batch_size):
                all_matches.extend(list(matches[j][1]))
            
            #print(time.time() -tic)
    all_scores = np.array(all_scores)
    all_matches = np.array(all_matches)
    sort_ids = np.argsort(all_scores)
    all_matches = all_matches[sort_ids[::-1]]

    metrics = {}
    AP, precisions, recalls, precision, recall = compute_ap(all_matches, gts, preds)
    metrics['AP'] = AP
    metrics['Precision'] = precision
    metrics['Recall'] = recall
    metrics['Forward Pass Time'] = t_fwd/len(loader.dataset)
    metrics['Postprocess Time'] = t_nms/len(loader.dataset) 

    cls_loss = cls_loss / len(loader)
    loc_loss = loc_loss / len(loader)
    metrics['loss'] = cls_loss + loc_loss

    return metrics, precisions, recalls, log_images


def eval_dataset(config, net, loss_fn, loader, device, e_range='all'):
    net.eval()
    if config['mGPUs']:
        net.module.set_decode(True)
    else:
        net.set_decode(True)

    t_fwds = 0
    t_post = 0
    loss_sum = 0

    img_list = range(len(loader.dataset))
    if e_range != 'all':
        e_range = min(e_range, len(loader.dataset))
        img_list = random.sample(img_list, e_range)

    log_img_list = random.sample(img_list, 10)

    gts = 0
    preds = 0
    all_scores = []
    all_matches = []
    log_images = []

    with torch.no_grad():
        for image_id in img_list:
            #tic = time.time()
            num_gt, num_pred, scores, pred_image, pred_match, loss, t_forward, t_nms = \
                eval_one(net, loss_fn, config, loader, image_id, device, plot=False)
            gts += num_gt
            preds += num_pred
            loss_sum += loss
            all_scores.extend(list(scores))
            all_matches.extend(list(pred_match))

            t_fwds += t_forward
            t_post += t_nms

            if image_id in log_img_list:
                log_images.append(pred_image)
            #print(time.time() - tic)
            
    all_scores = np.array(all_scores)
    all_matches = np.array(all_matches)
    sort_ids = np.argsort(all_scores)
    all_matches = all_matches[sort_ids[::-1]]

    metrics = {}
    AP, precisions, recalls, precision, recall = compute_ap(all_matches, gts, preds)
    metrics['AP'] = AP
    metrics['Precision'] = precision
    metrics['Recall'] = recall
    metrics['loss'] = loss_sum / len(img_list)
    metrics['Forward Pass Time'] = t_fwds / len(img_list)
    metrics['Postprocess Time'] = t_post / len(img_list)

    return metrics, precisions, recalls, log_images


def validation_round(net, device, epoch_num, writer, writer_counter):
    # Load Hyperparameters
    config, _, batch_size, _ = load_config()

    _, loss_fn = build_model(device, train=False)

    # Dataset and DataLoader
    _, test_data_loader, _, num_test = get_data_loader(frame_range=config['frame_range'])

    with torch.no_grad():
        ave_loss = 0

        with tqdm(total=num_test, desc='Validation: ',
                unit='pointclouds', leave=True, colour='green') as progress:

            for input, label_map, image_id in test_data_loader:
                input = input.to(device)
                label_map = label_map.to(device)

                # Forward
                predictions = net(input)
                loss, cls, loc = loss_fn(predictions, label_map)

                ave_loss += loss.item()

                # Update progress bar
                progress.set_postfix(
                    **{'loss': '{:.4f}'.format(abs(loss.item()))})
                progress.update(config['batch_size'])

                if progress.n % 50 == 0:
                        _, label_list = test_data_loader.dataset.get_label(image_id[0])

                        input_np = input[0].detach().cpu().permute(1, 2, 0).numpy()
                        corners, _ = filter_pred(predictions[0].detach().cpu())

                        plot_bev(input_np, label_list, window_name='GT',
                                 save_path='/home/aaron/PIXOR/viz/train/test/epoch{}_{}.jpg'.format(writer_counter, epoch_num))
                        plot_bev(input_np, corners, window_name='Prediction',
                                 save_path='/home/aaron/PIXOR/viz/train/test/epoch{}_{}.jpg'.format(writer_counter, epoch_num))


                # Add loss info to the Tensorboard logger
                writer.add_scalars(
                    main_tag='testing',
                    tag_scalar_dict={
                        'loss': loss.item(),
                        'cls_loss': cls,
                        'loc_loss': loc,
                    }, global_step=writer_counter)
                writer_counter += 1

        ave_loss = ave_loss / num_test

        if epoch_num == 0:
            print('Initial Benchmark Validation Loss: {:.5f}\n'.format(
                ave_loss))
        else:
            print('Validation Loss After Epoch {}: {:.5f}\n'.format(
                epoch_num, ave_loss))

        return ave_loss, writer_counter


def train(device):
    # Load Hyperparameters
    config, _, _, max_epochs = load_config()

    train_data_loader, _, num_train, _ = get_data_loader(frame_range=config['frame_range'])

    net, loss_fn, optimizer, scheduler = build_model(device, train=True)

    # Tensorboard Logger
    writer = SummaryWriter(log_dir=os.path.join(base_dir, 'log'))
    writer_counter = 0

    # net.eval()
    # images, _, _, _ = next(iter(train_data_loader))
    # writer.add_graph(net, images.to(device))

    if config['resume_training']:
        saved_ckpt_path = get_model_name(config)
        if config['mGPUs']:
            net.module.load_state_dict(torch.load(saved_ckpt_path, map_location=device))
        else:
            net.load_state_dict(torch.load(saved_ckpt_path, map_location=device))
        print('Successfully loaded trained ckpt at {}'.format(saved_ckpt_path))
        start_epoch = config['resume_from']
    else:
        start_epoch = 0

    for epoch in range(start_epoch, max_epochs):        
        epoch_loss = 0

        net.train()
        if config['mGPUs']:
            net.module.set_decode(True)
        else:
            net.set_decode(True)

        with tqdm(total=num_train, desc='Epoch %s/%s' % (epoch, max_epochs),
                  unit='pointclouds', leave=True, colour='green') as progress:

            for input, label_map, image_id in train_data_loader:
                input = input.to(device)
                label_map = label_map.to(device)
                optimizer.zero_grad()

                # Forward
                predictions = net(input)

                loss, cls, loc = loss_fn(predictions, label_map)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # Update progress bar
                progress.set_postfix(
                    **{'loss': '{:.4f}'.format(abs(loss.item()))})
                progress.update(config['batch_size'])

                with torch.no_grad():
                    if progress.n % 50 == 0:
                        _, label_list = train_data_loader.dataset.get_label(image_id[0])

                        input_np = input[0].detach().cpu().permute(1, 2, 0).numpy()
                        corners, _ = filter_pred(predictions[0].detach().cpu())

                        plot_bev(input_np, label_list, window_name='GT',
                                 save_path='/home/aaron/PIXOR/viz/train/truth/epoch{}_{}.jpg'.format(writer_counter, epoch))
                        plot_bev(input_np, corners, window_name='Prediction',
                                 save_path='/home/aaron/PIXOR/viz/train/output/epoch{}_{}.jpg'.format(writer_counter, epoch))

                    # Add loss info to the Tensorboard logger
                    writer.add_scalars(
                        main_tag='training',
                        tag_scalar_dict={
                            'loss': loss.item(),
                            'cls_loss': cls,
                            'loc_loss': loc,
                        }, global_step=writer_counter)
                    writer_counter += 1

        # Record Training Loss
        epoch_loss = epoch_loss / len(train_data_loader)

        # Run Validation
        ave_loss, writer_counter = validation_round(net, device, epoch, writer, writer_counter)

        # Save Checkpoint
        if (epoch + 1) == max_epochs or (epoch + 1) % config['save_every'] == 0:
            model_path = get_model_name(config, epoch + 1)
            if config['mGPUs']:
                torch.save(net.module.state_dict(), model_path)
            else:
                torch.save(net.state_dict(), model_path)
            print('Checkpoint saved at {}'.format(model_path))
        
        scheduler.step()

    print('Finished Training')


def eval_one(net, loss_fn, config, loader, image_id, device, plot=False, verbose=False):
    input, label_map, image_id = loader.dataset[image_id]
    input = input.to(device)
    label_map, label_list = loader.dataset.get_label(image_id)
    loader.dataset.reg_target_transform(label_map)
    label_map = torch.from_numpy(label_map).permute(2, 0, 1).unsqueeze_(0).to(device)

    # Forward Pass
    t_start = time.time()
    pred = net(input.unsqueeze(0))
    t_forward = time.time() - t_start

    loss, cls_loss, loc_loss = loss_fn(pred, label_map)
    pred.squeeze_(0)
    cls_pred = pred[0, ...]

    if verbose:
        print('Forward pass time', t_forward)


    # Filter Predictions
    t_start = time.time()
    corners, scores = filter_pred(pred)
    t_post = time.time() - t_start

    if verbose:
        print('Non max suppression time:', t_post)

    gt_boxes = np.array(label_list)
    gt_match, pred_match, overlaps = compute_matches(gt_boxes,
                                        corners, scores, iou_threshold=0.5)

    num_gt = len(label_list)
    num_pred = len(scores)
    input_np = input.cpu().permute(1, 2, 0).numpy()
    pred_image = get_bev(input_np, corners)

    if plot == True:
        # Visualization
        plot_bev(input_np, label_list, window_name='GT', save_path='/home/aaron/PIXOR/gt.jpg')
        plot_bev(input_np, corners, window_name='Prediction', save_path='/home/aaron/PIXOR/pred.jpg')
        plot_label_map(cls_pred.cpu().numpy())

    return num_gt, num_pred, scores, pred_image, pred_match, loss.item(), t_forward, t_post


def experiment(device, eval_range='all', plot=True):
    config, _, _, _ = load_config()
    net, loss_fn = build_model(device, train=False)
    state_dict = torch.load(get_model_name(config), map_location=device)
    if config['mGPUs']:
        net.module.load_state_dict(state_dict)
    else:
        net.load_state_dict(state_dict)
    train_loader, val_loader = get_data_loader(frame_range=config['frame_range'])

    #Train Set
    train_metrics, train_precisions, train_recalls, _ = eval_batch(config, net, loss_fn, train_loader, device, eval_range)
    print('Training mAP', train_metrics['AP'])
    fig_name = 'PRCurve_train_' + config['name']
    legend = 'AP={:.1%} @IOU=0.5'.format(train_metrics['AP'])
    plot_pr_curve(train_precisions, train_recalls, legend, name=fig_name)

    # Val Set
    val_metrics, val_precisions, val_recalls, _ = eval_batch(config, net, loss_fn, val_loader, device, eval_range)

    print('Validation mAP', val_metrics['AP'])
    print('Net Fwd Pass Time on average {:.4f}s'.format(val_metrics['Forward Pass Time']))
    print('Nms Time on average {:.4f}s'.format(val_metrics['Postprocess Time']))

    fig_name = 'PRCurve_val_' + config['name']
    legend = 'AP={:.1%} @IOU=0.5'.format(val_metrics['AP'])
    plot_pr_curve(val_precisions, val_recalls, legend, name=fig_name)


def test(device, image_id):
    config, _, _, _ = load_config()
    net, loss_fn = build_model(device, train=False)
    net.load_state_dict(torch.load(get_model_name(config), map_location=device))
    net.set_decode(True)
    train_loader, val_loader, _, _ = get_data_loader(frame_range=config['frame_range'])
    net.eval()

    with torch.no_grad():
        num_gt, num_pred, scores, pred_image, pred_match, loss, t_forward, t_nms = \
            eval_one(net, loss_fn, config, train_loader, image_id, device, plot=True)

        TP = (pred_match != -1).sum()
        print('Loss: {:.4f}'.format(loss))
        print('Precision: {:.2f}'.format(TP/num_pred))
        print('Recall: {:.2f}'.format(TP/num_gt))
        print('forward pass time {:.3f}s'.format(t_forward))
        print('nms time {:.3f}s'.format(t_nms))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PIXOR custom implementation')
    parser.add_argument('mode', choices=['train', 'val', 'test'], help='name of the experiment')
    parser.add_argument('--eval_range', type=int, help='range of evaluation')
    parser.add_argument('--test_id', type=int, default=25, help='id of the image to test')
    args = parser.parse_args()

    # Choose a device for the model
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    if args.mode == 'train':
        train(device)
    if args.mode == 'val':
        if args.eval_range is None:
            args.eval_range = 'all'
        experiment(device, eval_range=args.eval_range, plot=False)
    if args.mode == 'test':
        test(device, image_id=args.test_id)
