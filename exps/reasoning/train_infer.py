import argparse
import time

import os
import torch
from tensorboardX import SummaryWriter

from cvpack.torch_modeling.engine.engine import Engine
from cvpack.utils.pyt_utils import ensure_dir

from config import cfg
from model.infernet_C import InferNet
from lib.utils.dataloader_infer import get_train_loader
from lib.utils.solver import make_lr_scheduler, make_optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--SMAP_path", "-p", type=str, default='log/SMAP.pth',
                        help='Path to SMAP model')
    parser.add_argument("--batch_size", type=int, default=1,
                        help='Batch_size of test')

    with Engine(cfg, custom_parser=parser) as engine:
        logger = engine.setup_log(
            name='train', log_dir=cfg.OUTPUT_DIR, file_name='log.txt')
        args = engine.args
        ensure_dir(cfg.OUTPUT_DIR)

        model_infer = InferNet(cfg, run_efficient=cfg.RUN_EFFICIENT)

        device = torch.device(cfg.MODEL.DEVICE)
        model_infer.to(device)

        num_gpu = len(engine.devices)
        print(num_gpu) 
        #  default num_gpu: 8, adjust iter settings
        cfg.SOLVER.CHECKPOINT_PERIOD = int(cfg.SOLVER.CHECKPOINT_PERIOD * 8 / num_gpu)
        cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER * 8 / num_gpu)

        optimizer = make_optimizer(cfg, model_infer, num_gpu)
        scheduler = make_lr_scheduler(cfg, optimizer)

        engine.register_state(
            scheduler=scheduler, model=model_infer, optimizer=optimizer)

        if engine.distributed:
            model_infer = torch.nn.parallel.DistributedDataParallel(
                model_infer, device_ids=[args.local_rank],
                broadcast_buffers=False)


        data_loader = get_train_loader(cfg, num_gpu=num_gpu, is_dist=engine.distributed,
                                       use_augmentation=True, with_mds=cfg.WITH_MDS)
    
        # -------------------- do training -------------------- #
        logger.info("\n\nStart training with pytorch version {}".format(
            torch.__version__))

        max_iter = len(data_loader)
        checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
        if engine.local_rank == 0:
            tb_writer = SummaryWriter(cfg.TENSORBOARD_DIR)

        model_infer.train()

        time1 = time.time()
        for iteration, (heatmaps, valids, labels) in enumerate(
                data_loader, engine.state.iteration):
            iteration = iteration + 1
            heatmaps = heatmaps.to(device)
            valids = valids.to(device)
            labels = labels.to(device)
            
            loss_dict = model_infer(heatmaps, valids, labels, iter_num=iteration, max_iter=max_iter)
           
            losses = loss_dict['total_loss']

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            scheduler.step()


            if cfg.RUN_EFFICIENT:
                del heatmaps, valids, labels, losses

            if engine.local_rank == 0:
                if iteration % 20 == 0 or iteration == max_iter:
                    log_str = 'Max_iter:%d, Iter:%d, LR:%.1e, ' % (max_iter, 
                        iteration, optimizer.param_groups[0]["lr"] / num_gpu)
                    for key in loss_dict:
                        tb_writer.add_scalar(
                            key, loss_dict[key].mean(), global_step=iteration)
                        log_str += key + ': %.3f, ' % float(loss_dict[key])

                    time2 = time.time()
                    elapsed_time = time2 - time1
                    time1 = time2
                    required_time = elapsed_time / 20 * (max_iter - iteration)
                    hours = required_time // 3600
                    mins = required_time % 3600 // 60
                    log_str += 'To Finish: %dh%dmin,' % (hours, mins) 

                    logger.info(log_str)

            if iteration % checkpoint_period == 0 or iteration == max_iter:
                engine.update_iteration(iteration)
                if not engine.distributed or engine.local_rank == 0:
                    engine.save_and_link_checkpoint(cfg.OUTPUT_DIR)

            if iteration >= max_iter:
                logger.info('Finish training process!')
                break


if __name__ == "__main__":
    main()
