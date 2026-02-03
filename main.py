import json
import logging.config
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from configuration import config
from utils.data_loader import get_test_datalist, get_statistics
from utils.data_loader import get_train_datalist
from utils.method_manager import select_method

def save_plot(eval_results, args):
    """Helper function to create directories and save plots and results."""
    # Create results directory structure
    results_dir = "results"
    plots_dir = os.path.join(results_dir, "plots")
    dictionary_dir = os.path.join(results_dir, "dictionary")
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(dictionary_dir, exist_ok=True)
    
    # Prepare filename components
    dataset = args.dataset
    temp_batchsize = args.temp_batchsize if args.temp_batchsize else "None"
    score_threshold = getattr(args, 'score_threshold', None)
    score_threshold_str = str(score_threshold) if score_threshold is not None else "None"
    
    # Save eval_results as JSON
    json_filename = f"{dataset}_{temp_batchsize}_{score_threshold_str}.json"
    json_path = os.path.join(dictionary_dir, json_filename)
    
    # Convert defaultdict to regular dict for JSON serialization
    eval_results_dict = {k: v for k, v in eval_results.items()}
    with open(json_path, 'w') as f:
        json.dump(eval_results_dict, f, indent=2)
    
    return plots_dir, dataset, temp_batchsize, score_threshold_str


def create_plot(eval_results, args):
    """Creates plots of avg_mAP50 per domain, where sample_count is the x-axis."""
    plots_dir, dataset, temp_batchsize, score_threshold_str = save_plot(eval_results, args)
    
    # Extract data
    sample_counts = eval_results["data_cnt"]
    classwise_mAP50 = eval_results["classwise_mAP50"]
    
    # Determine number of domains (assuming all rows have same length)
    if len(classwise_mAP50) == 0:
        return
    
    num_domains = len(classwise_mAP50[0])
    
    # Create a plot for each domain
    for domain_idx in range(num_domains):
        # Extract mAP50 values for this domain across all evaluation points
        domain_mAP50s = [row[domain_idx] if domain_idx < len(row) else 0.0 
                         for row in classwise_mAP50]
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(sample_counts, domain_mAP50s, marker='o', linestyle='-')
        plt.xlabel('Sample Count')
        plt.ylabel('avg_mAP50')
        plt.title(f'Domain {domain_idx} - avg_mAP50 over Sample Count')
        plt.grid(True, alpha=0.3)
        
        # Determine domain label (source if domain_idx is 0, otherwise domain index)
        domain_label = 'source' if domain_idx == 0 else str(domain_idx)
        
        # Save the plot
        plot_filename = f"{domain_label}_{dataset}_{temp_batchsize}_{score_threshold_str}.png"
        plot_path = os.path.join(plots_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

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
        if getattr(args, 'gpu_transform', False):
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
    eval_dict = method.online_evaluate(samples_cnt, 0)
    eval_results["avg_mAP50"].append(eval_dict['avg_mAP50'])
    eval_results["classwise_mAP50"].append(eval_dict['classwise_mAP50'])
    eval_results["data_cnt"].append(samples_cnt)
    # breakpoint()
    method.save(samples_cnt, args.output_dir)
    for i in range(4): 
        data = train_datalist[i]
        #data in enumerate(train_datalist):
        samples_cnt += 1
        method.online_step(data, samples_cnt, args.n_worker)
        
        if samples_cnt % args.save_period == 0:
            method.save(samples_cnt, args.output_dir)
            
        #if samples_cnt % args.eval_period == 0:
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
    
    # Create and save plots
    create_plot(eval_results, args)

if __name__ == "__main__":
    main()