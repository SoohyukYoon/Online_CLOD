import logging.config
import os
import random
from collections import defaultdict

import numpy as np
import torch

from configuration import config
from utils.data_loader import get_test_datalist, get_statistics
from utils.data_loader import get_train_datalist
from utils.method_manager import select_method

def main():
    args = config.base_parser()

    logging.config.fileConfig("./configuration/logging.conf")
    logger = logging.getLogger()

    os.makedirs(f"results/{args.dataset}/{args.note}", exist_ok=True)
    os.makedirs(f"tensorboard/{args.dataset}/{args.note}", exist_ok=True)
    fileHandler = logging.FileHandler(f'results/{args.dataset}/{args.note}/seed_{args.rnd_seed}.log', mode="w")

    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    logger.info(args)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        if args.gpu_transform:
            args.gpu_transform = False
            logger.warning("Augmentation on GPU not available!")
    logger.info(f"Set the device ({device})")

    # Fix the random seeds
    torch.manual_seed(args.rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)

    # Transform Definition
    n_classes, image_path, label_path = get_statistics(dataset=args.dataset)

    logger.info(f"Select a method ({args.mode})")
    method = select_method(
        args, n_classes, device
    )

    eval_results = defaultdict(list)

    samples_cnt = 0
    task_id = 0

    # get datalist
    train_datalist = get_train_datalist(args.dataset, args.sigma, args.repeat, args.rnd_seed)

    method.n_samples(len(train_datalist))

    # Reduce datalist in Debug mode
    if args.debug:
        random.shuffle(train_datalist)
        train_datalist = train_datalist[:5000]

    print(f"total train stream: {len(train_datalist)}")
    # eval_dict = method.online_evaluate(samples_cnt, 0)
    # breakpoint()
    method.save(samples_cnt, args.output_dir)
    for i, data in enumerate(train_datalist):
        samples_cnt += 1
        method.online_step(data, samples_cnt, args.n_worker)
        
        if samples_cnt % args.save_period == 0:
            method.save(samples_cnt, args.output_dir)
            
        if samples_cnt % args.eval_period == 0:
            eval_dict = method.online_evaluate(samples_cnt, data["time"])
            eval_results["avg_mAP50"].append(eval_dict['avg_mAP50'])
            eval_results["classwise_mAP50"].append(eval_dict['classwise_mAP50'])
            eval_results["data_cnt"].append(samples_cnt)
            
    if eval_results["data_cnt"][-1] != samples_cnt:
        eval_dict = method.online_evaluate(samples_cnt, data["time"])

    A_last = eval_dict['avg_mAP50']

    # Accuracy (A)
    A_auc = np.mean(eval_results["avg_mAP50"])

    logger.info(f"======== Summary =======")
    logger.info(f"A_auc {A_auc} |  A_last {A_last} ") #| Total_flops {Total_flops}")

if __name__ == "__main__":
    main()