# Copyright (c) 2025, Biao Zhang.

import math
import sys
from typing import Iterable

from numpy import inner
import torch
import torch.nn.functional as F

import utils.misc as misc
import utils.lr_sched as lr_sched


def calc_iou(output, labels, threshold):
    target = torch.zeros_like(labels)
    target[labels>=threshold] = 1
    
    pred = torch.zeros_like(output)
    pred[output>=threshold] = 1

    accuracy = (pred==target).float().sum(dim=1) / target.shape[1]
    accuracy = accuracy.mean()
    intersection = (pred * target).sum(dim=1)
    union = (pred + target).gt(0).sum(dim=1) + 1e-5
    iou = intersection * 1.0 / union
    iou = iou.mean()
    return iou

def points_gradient(inputs, outputs):
    d_points = torch.ones_like(
        outputs, requires_grad=False, device=outputs.device)
    points_grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    
    # criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = torch.nn.L1Loss()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (points, labels, surface, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        surface = surface.to(device, non_blocking=True)
        # surface_normals = surface_normals.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
            points = points.requires_grad_(True)
            points_all = torch.cat([points, surface], dim=1)
            outputs = model(surface, points_all)

            output = outputs['o']

            grad = points_gradient(points_all, output)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
                # TODO: hard coded point numbers
                loss_eikonal = (grad[:, :].norm(2, dim=-1) - 1).pow(2).mean()
                loss_vol = criterion(output[:, :1024], labels[:, :1024])
                loss_near = criterion(output[:, 1024:2048], labels[:, 1024:2048])
                loss_surface = (output[:, 2048:]).abs().mean()
                
                # print(grad.shape, surface_normals.shape)
                # inner = torch.einsum('b n c, b n c -> b n', grad[:, 2048:], surface_normals)
                
                # print(inner.max(), inner.min(), inner.mean())
                # print(F.l1_loss(grad[:, 2048:], surface_normals), F.l1_loss(grad[:, 2048:], -surface_normals))
                # loss_surface_normal = F.l1_loss(F.normalize(grad[:, 2048:], dim=2), surface_normals)
                # loss_surface_normal = 1 - torch.einsum('b n c, b n c -> b n', (F.normalize(grad[:, 2048:], dim=2, eps=1e-6), surface_normals)).mean()

                loss = loss_vol + 10 * loss_near + 0.001 * loss_eikonal + 1 * loss_surface# + 0.01 * loss_surface_normal


        loss_value = loss.item()

        threshold = 0

        vol_iou = calc_iou(output[:, :1024], labels[:, :1024], threshold)
        near_iou = calc_iou(output[:, 1024:2048], labels[:, 1024:2048], threshold)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        metric_logger.update(loss_vol=loss_vol.item())
        metric_logger.update(loss_near=loss_near.item())
        metric_logger.update(loss_eikonal=loss_eikonal.item())
        metric_logger.update(loss_surface=loss_surface.item())
        # metric_logger.update(loss_surface_normal=loss_surface_normal.item())

        metric_logger.update(vol_iou=vol_iou.item())
        metric_logger.update(near_iou=near_iou.item())

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
