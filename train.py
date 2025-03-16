import argparse
import os
import torch
import json
from utils import set_seeds, get_logger
from model import HalluRLRAG
from data import get_dataloader
import sys


def train(args, model, train_dataloader, logger):
    optimizer = torch.optim.Adam([p for p in model.retriver.parameters() if p.requires_grad], lr=args.lr)

    total_reward_history = []
    total_loss_history = []

    for epoch in range(1, args.epochs + 1):
        logger.info("=========================================")
        logger.info(f"==== Epoch: {epoch} / {args.epochs} ====")

        train_reward = 0
        train_loss = 0.0

        # We can simply set the batch_size to len(train_data) in few-shot setting.
        for batch in train_dataloader:
            questions, answers, contexts = batch

            reward, loss = model(questions, answers, contexts)
            sys.exit()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_reward += reward
            train_loss += loss.item()

        total_reward_history.append(train_reward)
        total_loss_history.append(train_loss)

        best_reward = max(total_reward_history)
        best_loss = min(total_loss_history)
        best_reward_epoch = total_reward_history.index(best_reward)
        best_loss_epoch = total_loss_history.index(best_loss)

        logger.info(f"==== Total Reward for Epoch {epoch}: {train_reward} ====")
        logger.info(f"==== Total Loss for Epoch {epoch}: {train_loss} ====")
        logger.info(f"==== Best Reward: {best_reward} at Epoch {best_reward_epoch} ====")
        logger.info(f"==== Best Loss: {best_loss} at Epoch {best_loss_epoch} ====")

        # save every epoch
        ckpt = os.path.join(args.output_path, f"ckpt_{epoch}.pt")
        torch.save(model.retriver.linear.state_dict(), ckpt)
        logger.info(f"==== Save ckpt to {ckpt} ====")

        # save best epoch
        if epoch == best_reward_epoch:
            ckpt = os.path.join(args.output_path, "ckpt_best_reward.pt")
            torch.save(model.retriver.linear.state_dict(), ckpt)
            logger.info(f"==== Save best reward ckpt to {ckpt} ====")

        if epoch == best_loss_epoch:
            ckpt = os.path.join(args.output_path, "ckpt_best_loss.pt")
            torch.save(model.retriver.linear.state_dict(), ckpt)
            logger.info(f"==== Save best loss ckpt to {ckpt} ====")

        # save reward and loss history
        history = {
            "total_reward_history": total_reward_history,
            "total_loss_history": total_loss_history,
        }
        history_file = os.path.join(args.log_path, "history.json")
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4, separators=(',', ': '))

    # save in the end
    ckpt = os.path.join(args.output_path, "ckpt_final.pt")
    torch.save(model.retriver.linear.state_dict(), ckpt)
    logger.info(f"==== Save final ckpt to {ckpt} ====")


def get_args():
    parser = argparse.ArgumentParser()

    # retriver settings
    parser.add_argument(
        '--retriver_name', type=str, default='bert-base-uncased',
        choices=['distilbert-base-uncased', 'bert-base-uncased'],
        help='base model for retriver'
    )
    parser.add_argument(
        '--embedding_size', type=int, default=128, 
        help='Hidden state size of final layer in retriver'
    )

    # base model settings
    parser.add_argument(
        '--base_model_name', type=str, default='deepseek-v3-671b', 
        help='base model for generating answers'
    )

    # evaluator settings
    parser.add_argument(
        '--evaluator_name', type=str, default='deepseek-v3-671b', 
        help='base model for evaluator'
    )
    parser.add_argument(
        '--temperature', type=float, default=0.0,
        help='temperature for base model'
    )
    parser.add_argument(
        '--max_tokens', type=int, default=128,
        help='The maximum number of tokens allowed for the generated answer.'
    )
    parser.add_argument(
        '--top_p', type=float, default=1.0,
        help='The cumulative probability threshold for nucleus sampling.'
    )
    parser.add_argument(
        '--frequency_penalty', type=float, default=0.0,
        help='The penalty for repeating tokens.'
    )
    parser.add_argument(
        '--presence_penalty', type=float, default=0.0,
        help='The penalty for not repeating tokens.'
    )

    # training settings
    parser.add_argument(
        '--seed', type=int, default=1, 
        help='random seed'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='training device'
    )
    parser.add_argument(
        '--n_shot', type=int, default=2, 
        help='Number of n-shot training examples.'
    )
    parser.add_argument(
        '--n_samples', type=int, default=20,
        help='Number of training samples.'
    )
    parser.add_argument(
        '--n_candidates', type=int, default=10, 
        help='Number of candidate prompts.'
    )
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='Learning rate of policy network.'
    )
    parser.add_argument(
        '--epochs', type=int, default=20, 
        help='Number of training epochs.'
    )
    parser.add_argument(
        '--batch_size', type=int, default=2,
        help='Training batch size. Set to n_samples by default.'
    )

    # save settings
    parser.add_argument(
        '--log_path', type=str, default='./log',
        help='path to log th results'
    )
    parser.add_argument(
        '--output_path', type=str, default='./checkpoints',
        help='path to save the model'
    )

    args = parser.parse_args()

    # print and save the args
    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    return args


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
        args.frequency_penalty,
        args.presence_penalty,
        args.n_shot
    ).to(args.device)

    train_dataloader = get_dataloader(args, split='train')

    logger = get_logger(args)

    train(args, model, train_dataloader, logger)


if __name__ == '__main__':
    main()
