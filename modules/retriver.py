import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForTokenClassification
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

        self.model = AutoModelForTokenClassification.from_pretrained(retriver_name)
        self.tokenizer = AutoTokenizer.from_pretrained(retriver_name)

        # Freeze transformer encoder and only train the linear layer
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

    def forward(self, input_list, linear_forward=True):
        input_ids = self.tokenizer(input_list, truncation=True, padding=True, return_tensors="pt").to(self.model.device)
        output = self.model(**input_ids, output_hidden_states=True)

        # Get [CLS] hidden states of last hidden layer
        context_embedding = output.hidden_states[-1][:, 0, :] 

        if linear_forward and self.linear:
            context_embedding = self.linear(context_embedding)

        return context_embedding
