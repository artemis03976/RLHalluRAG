import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.retriver import PolicyRetriever
from modules.base import BaseModel
from modules.evaluator import HalluEvaluator
import sys


class HalluRLRAG(nn.Module):
    def __init__(
        self,
        # retriver settings
        retriver_name,
        embedding_size,
        # base model settings
        base_model_name,
        # evauator settings
        evaluator_name,
        # common
        temperature,
        max_tokens,
        top_p,
        frequency_penalty,
        presence_penalty,
        # misc
        n_shot
    ):
        super().__init__()

        self.n_shot = n_shot

        self.retriver = PolicyRetriever(
            retriver_name=retriver_name,
            embedding_size=embedding_size
        )

        self.base_model = BaseModel(
            base_model_name=base_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )

        self.evaluator = HalluEvaluator(
            evaluator_name=evaluator_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
    
    def select_context(self, contexts, scores):
        sample_prob = scores.clone().detach().squeeze().cpu().numpy()
        # replace np.nan with 0
        sample_prob = np.nan_to_num(sample_prob, nan=0.000001) 
        # normalize to 1
        sample_prob /= sample_prob.sum()
        # select most relative contexts
        idxs = np.random.choice(range(len(contexts)), self.n_shot, p=sample_prob, replace=False)
        # reverse shot_pids so more relevant prompt will be put closer to the question
        idxs = idxs[::-1]
        shot_contexts = [contexts[idx] for idx in idxs]

        return idxs, shot_contexts
    
    def get_reward_loss(self, selected_idxs, scores, judgements):
        batch_loss = 0
        batch_reward = 0

        for i in range(len(judgements)):
            log_prob = 0
            for selected_idx in selected_idxs[i]:
                log_prob += torch.log(scores[i][selected_idx])

            reward = 1 if judgements[i] == 'NO' else -1

            batch_reward += reward
            batch_loss -= reward * log_prob
        
        return batch_reward, batch_loss

    def forward(self, questions, answers, contexts):
        relative_contexts = []
        relative_contexts_idxs = []
        scores = []
        for i in range(len(questions)):
            embedding_questions = self.retriver(questions[i])  # 1 x embedding_size 
            embedding_contexts = self.retriver(contexts[i])  # n_contexts x embedding_size

            score = torch.mm(embedding_questions, embedding_contexts.t())  # 1 x n_contexts
            score = F.softmax(score, dim=1)  # 1 x n_contexts
            
            idxs, relative_context = self.select_context(contexts[i], score)

            scores.append(score.squeeze())
            relative_contexts.append(relative_context)
            relative_contexts_idxs.append(idxs)
       
        generated_answers = self.base_model(relative_contexts, questions)

        judgements = self.evaluator(questions, answers, generated_answers)

        return self.get_reward_loss(relative_contexts_idxs, scores, judgements)
