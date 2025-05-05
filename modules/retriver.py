import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import logging


loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    if "transformers" in logger.name.lower():
        logger.setLevel(logging.ERROR)


class PolicyRetriever(nn.Module):
    def __init__(
        self,
        retriver_name="bert-base-uncased",
        add_linear=True,
        embedding_size=128,
        freeze_encoder=True,
    ) -> None:
        super().__init__()

        self.model = AutoModel.from_pretrained(retriver_name)
        self.tokenizer = AutoTokenizer.from_pretrained(retriver_name)

        # Freeze encoder and only train the linear layer
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

        if add_linear:
            # Add an additional small, adjustable linear layer on top of BERT tuned through RL
            self.embedding_size = embedding_size
            input_dim = self.model.config.hidden_size  # 768 for bert-base-uncased, distilbert-base-uncased
            self.linear = nn.Linear(input_dim, embedding_size)
        else:
            self.linear = None

        self.to("cuda:0")

    def get_embedding(self, input_list):
        input_ids = self.tokenizer(input_list, truncation=True, padding=True, return_tensors="pt").to(self.model.device)
        output = self.model(**input_ids, output_hidden_states=True)

        # Get [CLS] hidden states of last hidden layer
        embedding = output.last_hidden_state[:, 0, :] 

        if self.linear:
            embedding = self.linear(embedding)

        return embedding

    def forward(self, batch_questions, batch_contexts):
        batch_size = len(batch_contexts)
        n_contexts = len(batch_contexts[0]) if batch_size > 0 else 0

        embedding_questions = self.get_embedding(batch_questions)  # [B, embedding_size]

        # Flatten batch
        flattened_batch_contexts = []
        for batch in batch_contexts:
            flattened_batch_contexts.extend(batch)
        embedding_contexts = self.get_embedding(flattened_batch_contexts).view(batch_size, n_contexts, -1)  # [B, n_preselect, embedding_size]

        batch_scores = torch.bmm(
            embedding_questions.unsqueeze(1), 
            embedding_contexts.transpose(1, 2)
        ).squeeze(1)  # [B, n_preselect]

        batch_probs = torch.softmax(batch_scores, dim=-1)
        batch_log_probs = torch.log_softmax(batch_scores, dim=-1)

        return batch_scores, batch_probs, batch_log_probs
