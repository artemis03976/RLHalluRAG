import os
import json
from torch.utils.data import Dataset, DataLoader
import sys


class HotPotQADataset(Dataset):
    def __init__(
            self, 
            root='./data/HotpotQA', 
            split='train',
        ):
        super().__init__()

        self.root = root
        self.split = split

        self.data_path = os.path.join(self.root, f'{self.split}.json')
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.data_ids = [item['_id'] for item in data]
        self.questions = [item['question'] for item in data]
        self.contexts = [[f"{context[0]}: {''.join(context[1])}" for context in item["context"]] for item in data]
        self.supporting_facts = [item['supporting_facts'] for item in data]

        if split != 'test':
            self.answers = [item['answer'] for item in data]
    
    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        return (self.questions[idx], self.answers[idx] if self.split != 'test' else None, self.contexts[idx])


def collate_fn(batch):
    questions = [item[0] for item in batch]
    answers = [item[1] for item in batch]
    contexts = [item[2] for item in batch]

    return (questions, answers, contexts)


def get_dataloader(args, split='train'):
    dataset = HotPotQADataset(split=split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    return dataloader


if __name__ == '__main__':
    dataset = HotPotQADataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    for batch in dataloader:
        print(len(batch[2]))
        sys.exit()
    