import torch
import argparse
from argparse import Namespace
import os
import json
from utils import set_seeds
from model import HalluRLRAG
from data import get_dataloader
from tqdm import tqdm


def inference(args, model, test_dataloader):
    with torch.no_grad():
        model.eval()
        test_acc = 0.0 
        for batch in tqdm(test_dataloader):
            outputs = model.generate(batch)
            test_acc += sum(outputs)
        print(f"Test Accuracy: {test_acc / args.n_samples}")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--checkpoint_path', type=str, default='./checkpoints',
        help='path to save the model'
    )
    parser.add_argument(
        '--n_samples', type=int, default=1000,
        help='Number of testing samples.'
    )

    args = parser.parse_args()

    with open(os.path.join(args.checkpoint_path, "config.json"), "r") as f:
        args_dict = json.load(f)
    base_args = Namespace(**args_dict)

    merged = vars(base_args).copy()
    merged.update(vars(args))

    return Namespace(**merged)


def main():
    args = get_args()

    set_seeds(args)

    model = HalluRLRAG (
        args.retriver_name,
        args.embedding_size,
        args.base_model_name,
        args.evaluator_name,
        args.temperature,
        args.max_tokens,
        args.top_p,
        args.method,
        args.use_hf_model,
        args.n_shot,
        args.n_preselect,
        args.n_gpus,
    )

    state_dict = torch.load(os.path.join(args.checkpoint_path, 'ckpt_best_reward.pt'))
    model.retriver.linear.load_state_dict(state_dict)

    test_dataloader = get_dataloader(args, name='NQ', split='dev')

    inference(args, model, test_dataloader)


if __name__ == '__main__':
    main()
