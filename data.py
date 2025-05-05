import os
import json
import random
from torch.utils.data import Dataset, DataLoader


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
        return {
            'questions': self.questions[idx],
            'contexts': self.contexts[idx],
            'answers': self.answers[idx] if self.split != 'test' else None
        }


class NQDataset(Dataset):
    def __init__(
            self,
            root='./data/NQ', 
            split='train',
            n_samples=1000,
    ):
        self.root = root
        self.split = split
        self.n_samples = n_samples

        self.data_path = os.path.join(self.root, f'{self.split}.json')
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)[:self.n_samples]
            # data = random.sample(json.load(f), self.n_samples)

        self.questions = [item['question'] for item in data]
        self.contexts = [[f"{context['title']}: {context['text']}" for context in item["ctxs"]] for item in data]
        if split != 'test':
            self.answers = [item['answers'] for item in data]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return {
            'questions': self.questions[idx],
            'contexts': self.contexts[idx],
            'answers': self.answers[idx] if self.split != 'test' else None
        }
        

class MultiDocQAChineseDataset(Dataset):
    def __init__(
            self, 
            root='./data/Multi-Doc-QA-Chinese', 
            split='train',
        ):
        super().__init__()

        self.root = root
        self.split = split

        self.data_path = os.path.join(self.root, f'{self.split}.json')


def collate_fn(batch):
    return {
        'questions': [item['questions'] for item in batch],
        'answers': [item['answers'] for item in batch],
        'contexts': [item['contexts'] for item in batch]
    }


def get_dataloader(args, name='HotPotQA', split='train'):
    if name == 'HotPotQA':
        dataset = HotPotQADataset(split=split)
    elif name == 'NQ':
        dataset = NQDataset(split=split, n_samples=args.n_samples)

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=8,
    )
    
    return dataloader


if __name__ == '__main__':
    dataset = NQDataset()
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    for batch in dataloader:
        # print(batch)
        print(batch)
        break
    