import argparse


def base_parser():
    parser = argparse.ArgumentParser(description="Class Incremental Learning Research")

    # Mode and Exp. Settings.
    parser.add_argument(
        "--mode",
        type=str,
        default="er",
        help="Select CIL method",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="[mnist, cifar10, cifar100, imagenet]",
    )

    parser.add_argument("--rnd_seed", type=int, help="Random seed number.")
    parser.add_argument(
        "--memory_size", type=int, default=500, help="Episodic memory size"
    )
    # Dataset
    parser.add_argument(
        "--log_path",
        type=str,
        default="results",
        help="The path logs are saved.",
    )
    parser.add_argument("--sigma", type=int, default=10, help="Sigma of gaussian*100")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times to repeat period")
    # Model
    parser.add_argument(
        "--model_name", type=str, default="resnet18", help="Model name"
    )
    parser.add_argument('--output-dir', type=str, default='online_CL_snapshot', help='directory where to save results')
    parser.add_argument('--save_period', type=int, default=100, help='period to save model')
    
    # Train
    parser.add_argument("--opt_name", type=str, default="sgd", help="Optimizer name")
    parser.add_argument("--sched_name", type=str, default="default", help="Scheduler name")
    parser.add_argument("--batchsize", type=int, default=16, help="batch size")

    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")

    parser.add_argument(
        "--use_amp", action="store_true", help="Use automatic mixed precision."
    )
    
    # CL_Loader
    parser.add_argument("--n_worker", type=int, default=4, help="The number of workers")
    parser.add_argument("--data_dir", type=str, help="location of the dataset")

    # Debug
    parser.add_argument("--debug", action="store_true", help="Turn on Debug mode")
    # Note
    parser.add_argument("--note", type=str, help="Short description of the exp")

    # Eval period
    parser.add_argument("--eval_period", type=int, default=100, help="evaluation period for true online setup")
    parser.add_argument("--temp_batchsize", type=int, help="temporary batch size, for true online")
    parser.add_argument("--online_iter", type=float, default=1, help="number of model updates per samples seen.")

    parser.add_argument("--selection_method", type=str, default="loss")
    parser.add_argument("--priority_selection", type=str, default="high")
    parser.add_argument("--unfreeze_rate", type=float, default=0.0)
    parser.add_argument("--fisher_ema_ratio", type=float, default=0.01)

    args = parser.parse_args()
    return args