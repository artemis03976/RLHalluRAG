import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.retriver import PolicyRetriever
from modules.base import BaseModel
from modules.evaluator import HalluEvaluator
from rank_bm25 import BM25Okapi


def bm25_sparse_retrieve(query, contexts, n_preselect):
    tokenized_contexts = []
    for context in contexts:
        context_tokens = context.split()
        tokenized_contexts.append(context_tokens)
    bm25 = BM25Okapi(tokenized_contexts)
    tokenized_query = query.split(" ")
    sparse_retrived_contexts = bm25.get_top_n(tokenized_query, contexts, n=n_preselect)

    return sparse_retrived_contexts


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
        # misc,
        method,
        use_hf_model,
        n_shot,
        n_preselect,
    ):
        super().__init__()

        self.method = method
        self.n_shot = n_shot
        self.n_preselect = n_preselect

        self.retriver = PolicyRetriever(
            retriver_name=retriver_name,
            embedding_size=embedding_size
        )

        self.base_model = BaseModel(
            base_model_name=base_model_name,
            use_hf_model=use_hf_model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        self.evaluator = HalluEvaluator(
            evaluator_name=evaluator_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
    
    def select_context(self, contexts, prob, n_shot):
        # select most relative contexts
        idxs = torch.multinomial(prob, n_shot, replacement=False)
        # reverse shot_pids so more relevant prompt will be put closer to the question
        # idxs = idxs[::-1]
        shot_contexts = [contexts[idx] for idx in idxs]

        return idxs, shot_contexts

    def forward_reinforce(self, batch):
        batch_loss = 0.0
        batch_reward = 0.0

        for sample in batch:
            question, answer, context = sample['questions'], sample['answers'], sample['contexts']
            # spaese retrive using BM25
            sparse_retrived_contexts = bm25_sparse_retrieve(question, context, self.n_preselect)
            # get retriver score 
            score, prob, log_prob = self.retriver(question, sparse_retrived_contexts)
            # get final retrived contexts
            k = min(self.n_shot, (prob > 1e-6).sum().item())
            idxs, relative_context = self.select_context(sparse_retrived_contexts, prob, k)

            # test from 0-shot to k-shot
            shot_loss, shot_reward = 0.0, 0.0
            for i in range(0, k + 1):
                generated_answer = self.base_model(relative_context[:i], question)

                judgement = self.evaluator(question, answer, generated_answer)
                # print(judgement)

                if judgement == 'NO':
                    reward = 1
                elif judgement == 'YES':
                    reward = -1
                else:
                    reward = 0

                if i != 0:
                    shot_loss -= reward * log_prob[idxs[i - 1]]
                shot_reward += reward
            
            batch_loss += shot_loss
            batch_reward += shot_reward

        return batch_reward / len(batch), batch_loss / len(batch)
    
    def forward_grpo(self, group):
        group_reward = []
        group_log_prob = []
        
        for sample in group:
            question, answer, context = sample['questions'], sample['answers'], sample['contexts']
            # spaese retrive using BM25
            sparse_retrived_contexts = bm25_sparse_retrieve(question, context, self.n_preselect)
            # get retriver score 
            score, prob, log_prob = self.retriver(question, sparse_retrived_contexts)
            # get final retrived contexts
            k = min(self.n_shot, (prob > 1e-6).sum().item())
            idxs, relative_context = self.select_context(sparse_retrived_contexts, prob, k)

            # test from 0-shot to k-shot
            rewards = []
            for i in range(0, k + 1):
                generated_answer = self.base_model(relative_context[:i], question)

                judgement = self.evaluator(question, answer, generated_answer)
                # print(judgement)

                if judgement == 'NO':
                    reward = 1
                elif judgement == 'YES':
                    reward = -1
                else:
                    reward = 0
                
                rewards.append(reward)

            group_log_prob.append(torch.sum(log_prob[idxs]))
            group_reward.append(np.mean(rewards))
            
        group_reward = torch.tensor(group_reward, dtype=log_prob.dtype, device=log_prob.device)
        group_log_prob = torch.stack(group_log_prob)

        # compute relative advantages
        mean_reward = group_reward.mean()
        std_reward = group_reward.std() + 1e-8
        relative_advantages = (group_reward - mean_reward) / std_reward

        # compute group loss
        group_loss = -torch.mean(group_log_prob * relative_advantages)
        
        return torch.mean(relative_advantages), group_loss
    
    def forward(self, batch):
        if self.method == 'reinforce':
            return self.forward_reinforce(batch)
        elif self.method == 'grpo':
            return self.forward_grpo(batch)

    def generate(self, batch):
        results = []

        for sample in batch:
            question, answer, context = sample['questions'], sample['answers'], sample['contexts']
            # spaese retrive using BM25
            sparse_retrived_contexts = bm25_sparse_retrieve(question, context, self.n_preselect)
            # get retriver score 
            score, prob, log_prob = self.retriver(question, sparse_retrived_contexts)
            # get final retrived contexts
            k = min(self.n_shot, (prob > 1e-6).sum().item())
            _, relative_context = self.select_context(sparse_retrived_contexts, prob, k)

            generated_answer = self.base_model(relative_context, question)

            judgement = self.evaluator(question, answer, generated_answer)

            if judgement == 'NO':
                acc = 1
            elif judgement == 'YES':
                acc = 0
            
            results.append(acc)
        
        return results
    
    def bm25_generate(self, batch):
        results = []

        for sample in batch:
            question, answer, context = sample['questions'], sample['answers'], sample['contexts']

            retrived_contexts = bm25_sparse_retrieve(question, context, self.n_shot)

            generated_answer = self.base_model(retrived_contexts, question)

            judgement = self.evaluator(question, answer, generated_answer)

            if judgement == 'NO':
                acc = 1
            elif judgement == 'YES':
                acc = 0
            
            results.append(acc)
        
        return results
