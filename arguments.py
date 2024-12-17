import torch
import numpy as np
import shutil
import sys
import math
import logging
import yaml
import argparse
import os


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    # arguments for the ddpg
    parser.add_argument(
        #"--config", type=str, required=True, help="Path to the config file"
        "--config", type=str, default='celeba_hq.yml', help="Path to the config file"  # e.g.,  celeba_hq, imagenet_256
    )
    parser.add_argument("--seed", type=int, default=1234, help="Set different seeds for diverse results")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        #"--deg", type=str, required=True, help="Degradation"
        "--deg", type=str, default='deblur_gauss', help="Degradation" # e.g., sr_bicubic, deblur_gauss
    )
    parser.add_argument(
        "--path_y",
        type=str,
        #required=True,
        default='celeba_hq', # e.g.,  celeba_hq, imagenet
        help="Path of the test dataset.",
    )
    parser.add_argument(
        "--sigma_y", type=float, default=0.05, help="sigma_y"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="demo",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--save_y", dest="save_observed_img", action="store_true"
    )    
    parser.add_argument(
        "--deg_scale", type=float, default=0.0, help="deg_scale"
    )    
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        "--ni",
        action="store_false",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        '--subset_start', type=int, default=-1
    )
    parser.add_argument(
        '--subset_end', type=int, default=-1
    )
    parser.add_argument(
        "--operator_imp", type=str, default="FFT", help="SVD | FFT"  # TODO: add CG support
    )
    parser.add_argument(
        "--scale_ls", type=float, default=1.0, help="scale_for_gLS"
    )
    parser.add_argument(
        "--inject_noise", type=int, default=1, help="inject_noise --- separates between DDPG and IDPG"
    )
    parser.add_argument(
        "--gamma", type=float, default=8.0, help="gamma parameterizes traversing from BP to LS, bigger means more dominance to BP"
    )
    parser.add_argument(
        "--xi", type=float, default=1e-5, help="xi -- obsolete, can be used for regularization instead of eta_tilde"
    )
    parser.add_argument(
        "--eta_tilde", type=float, default=0.7, help="eta_tilde regularizes pinv"
    )
    parser.add_argument(
        "--zeta", type=float, default=0.5, help="for inject_noise, zeta trades between effective estimated noise and random noise"
    )
    parser.add_argument(
        "--step_size_mode", type=int, default=1, help="0 (fixed 1) | 1 (certain decay as in paper) | 2 (fixed 1 for BP, decay for LS)" # you can add other choices
    )
    
    # arguments for the rl model
    parser.add_argument("--input_root", type=str, default="/disk_195a/qiannnhui", help="The root folder of input images")
    parser.add_argument("--second_stage", action="store_true", help="Whether to run the second stage")
    parser.add_argument("--RL_algorithm", type=str, default="SAC", help="The RL algorithm to use")
    parser.add_argument("--eval_model_name", type=str, default="new_SR_2agent_A2C_10")

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    with open(os.path.join("ddrm/configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    # tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
    args.image_folder = os.path.join(args.exp, "image_samples", args.image_folder)
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input(
                f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
            )
            if response.upper() == "Y":
                overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
