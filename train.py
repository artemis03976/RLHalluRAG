import argparse
import os
import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel
import json
from utils import set_seeds, get_logger
from model import HalluRLRAG
from data import get_dataloader
from tqdm import tqdm
from accelerate import Accelerator


def train_reinforce(args, model, train_dataloader, logger, accelerator):
    optimizer = torch.optim.Adam([p for p in model.retriver.parameters() if p.requires_grad], lr=args.lr)

    total_reward_history = []
    total_loss_history = []

    model, train_dataloader, optimizer = accelerator.prepare(model, train_dataloader, optimizer)

    for epoch in range(1, args.epochs + 1):
        if accelerator.is_main_process:
            logger.info("=========================================")
            logger.info(f"==== Epoch: {epoch} / {args.epochs} ====")

        train_reward = 0.0
        train_loss = 0.0

        for batch in tqdm(train_dataloader):
            reward, loss = model(batch)
    
            optimizer.zero_grad()
            accelerator.backward(loss)
            if isinstance(model, DistributedDataParallel):
                clip_grad_norm_(model.module.retriver.linear.parameters(), max_norm=1.0)
            else:
                clip_grad_norm_(model.retriver.linear.parameters(), max_norm=1.0)
            optimizer.step()

            train_reward += reward
            train_loss += loss.item()

        total_reward_history.append(train_reward)
        total_loss_history.append(train_loss)

        best_reward = max(total_reward_history)
        best_loss = min(total_loss_history)
        best_reward_epoch = total_reward_history.index(best_reward) + 1
        best_loss_epoch = total_loss_history.index(best_loss) + 1

        if accelerator.is_main_process:
            logger.info(f"==== Total Reward for Epoch {epoch}: {train_reward} ====")
            logger.info(f"==== Total Loss for Epoch {epoch}: {train_loss} ====")
            logger.info(f"==== Best Reward: {best_reward} at Epoch {best_reward_epoch} ====")
            logger.info(f"==== Best Loss: {best_loss} at Epoch {best_loss_epoch} ====")

            # save every epoch
            ckpt = os.path.join(args.output_path, f"ckpt_{epoch}.pt")
            if isinstance(model, DistributedDataParallel):
                torch.save(model.module.retriver.linear.state_dict(), ckpt)
            else:
                torch.save(model.retriver.linear.state_dict(), ckpt)
            logger.info(f"==== Save ckpt to {ckpt} ====")

            # save best epoch
            if epoch == best_reward_epoch:
                ckpt = os.path.join(args.output_path, "ckpt_best_reward.pt")
                if isinstance(model, DistributedDataParallel):
                    torch.save(model.module.retriver.linear.state_dict(), ckpt)
                else:
                    torch.save(model.retriver.linear.state_dict(), ckpt)
                logger.info(f"==== Save best reward ckpt to {ckpt} ====")

            if epoch == best_loss_epoch:
                ckpt = os.path.join(args.output_path, "ckpt_best_loss.pt")
                if isinstance(model, DistributedDataParallel):
                    torch.save(model.module.retriver.linear.state_dict(), ckpt)
                else:
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
        
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # save in the end
        ckpt = os.path.join(args.output_path, "ckpt_final.pt")
        if isinstance(model, DistributedDataParallel):
            torch.save(model.module.retriver.linear.state_dict(), ckpt)
        else:
            torch.save(model.retriver.linear.state_dict(), ckpt)
        logger.info(f"==== Save final ckpt to {ckpt} ====")
    
    accelerator.wait_for_everyone()


def train_grpo(args, model, train_dataloader, logger, accelerator):
    optimizer = torch.optim.Adam([p for p in model.retriver.parameters() if p.requires_grad], lr=args.lr)

    total_reward_history = []
    total_loss_history = []

    for epoch in range(1, args.epochs + 1):
        if accelerator.is_main_process:
            logger.info("=========================================")
            logger.info(f"==== Epoch: {epoch} / {args.epochs} ====")

        train_reward = 0.0
        train_loss = 0.0

        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()

            # split batch by group size
            groups = [batch[i: i + args.group_size] for i in range(0, len(batch), args.group_size)]
            group_loss = 0.0
            group_reward = 0.0

            for group in groups:
                reward, loss = model(group)

                accelerator.backward(loss)
                group_loss += loss.item()
                group_reward += reward.item()

            clip_grad_norm_(model.retriver.linear.parameters(), max_norm=1.0)
            optimizer.step()

            train_reward += group_reward / len(groups)
            train_loss += group_loss / len(groups)
        
        total_reward_history.append(train_reward)
        total_loss_history.append(train_loss)

        best_reward = max(total_reward_history)
        best_loss = min(total_loss_history)
        best_reward_epoch = total_reward_history.index(best_reward) + 1
        best_loss_epoch = total_loss_history.index(best_loss) + 1

        if accelerator.is_main_process:
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
        
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # save in the end
        ckpt = os.path.join(args.output_path, "ckpt_final.pt")
        torch.save(model.retriver.linear.state_dict(), ckpt)
        logger.info(f"==== Save final ckpt to {ckpt} ====")

    accelerator.wait_for_everyone()


def train(args, model, train_dataloader, logger, accelerator):
    if args.method == 'reinforce':
        train_reinforce(args, model, train_dataloader, logger, accelerator)
    elif args.method == 'grpo':
        train_grpo(args, model, train_dataloader, logger, accelerator)


def get_args():
    parser = argparse.ArgumentParser()

    # retriver settings
    parser.add_argument(
        '--retriver_name', type=str, default='bert-base-uncased',
        choices=['distilbert-base-uncased', 'bert-base-uncased'],
        help='base model for retriver'
    )
    parser.add_argument(
        '--n_preselect', type=int, default=20, 
        help='Number of preselected contexts.'
    )
    parser.add_argument(
        '--embedding_size', type=int, default=256, 
        help='Hidden state size of final layer in retriver'
    )

    # base model settings
    parser.add_argument(
        '--base_model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct', 
        help='base model for generating answers'
    )
    parser.add_argument(
        '--use_hf_model', action='store_true', default=False, 
        help='use huggingface models'
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

    # evaluator settings
    parser.add_argument(
        '--evaluator_name', type=str, default='deepseek-v3-671b', 
        help='base model for evaluator'
    )

    # training settings
    parser.add_argument(
        '--seed', type=int, default=42, 
        help='random seed'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='training device'
    )
    parser.add_argument(
        '--method', type=str, default='reinforce',
        choices=['reinforce', 'grpo'],
        help='training method'
    )
    parser.add_argument(
        '--n_shot', type=int, default=5, 
        help='Number of n-shot training examples.'
    )
    parser.add_argument(
        '--n_samples', type=int, default=1000,
        help='Number of training samples.'
    )
    parser.add_argument(
        '--lr', type=float, default=0.0001,
        help='Learning rate of policy network.'
    )
    parser.add_argument(
        '--epochs', type=int, default=5, 
        help='Number of training epochs.'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Training batch size.'
    )
    parser.add_argument(
        '--group_size', type=int, default=2,
        help='Number of training samples.'
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

    accelerator = Accelerator(
        mixed_precision='bf16'
    )

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
    ).to(accelerator.device)

    train_dataloader = get_dataloader(args, name='NQ', split='train')

    logger = get_logger(args)

    train(args, model, train_dataloader, logger, accelerator)


if __name__ == '__main__':
    main()
