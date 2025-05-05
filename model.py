import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.retriver import PolicyRetriever
from modules.base import BaseModel
from modules.evaluator import HalluEvaluator
from rank_bm25 import BM25Okapi
import sys


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
        n_gpus,
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
            n_gpus=n_gpus,
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
        batch_size = len(batch['questions'])

        batch_questions = batch['questions']
        batch_answers = batch['answers']
        batch_contexts = batch['contexts']

        batch_loss = 0.0
        batch_reward = 0.0

        # spaese retrive using BM25
        sparse_retrived_contexts = [
            bm25_sparse_retrieve(q, ctx, self.n_preselect)
            for q, ctx in zip(batch_questions, batch_contexts)
        ]

        # get retriver score 
        scores, probs, log_probs = self.retriver(batch_questions, sparse_retrived_contexts)
        # get final retrived contexts
        batch_idxs, batch_relative_contexts = [], []
        for i in range(batch_size):
            k = min(self.n_shot, (probs[i] > 1e-6).sum().item())
            idxs, relative_context = self.select_context(sparse_retrived_contexts[i], probs[i], k)
            batch_idxs.append(idxs)
            batch_relative_contexts.append(relative_context)

        # test from 0-shot to k-shot
        for i in range(batch_size):
            idxs = batch_idxs[i]
            contexts = batch_relative_contexts[i]
            shot_contexts = [contexts[:j] for j in range(len(idxs) + 1)]

            batch_generated_answers = self.base_model(shot_contexts, batch_questions[i])

            judgements = self.evaluator(
                [batch_questions[i] for _ in range(len(idxs) + 1)],
                [batch_answers[i] for _ in range(len(idxs) + 1)], 
                batch_generated_answers
            )

            reward = torch.tensor(
                [1 if item == "NO" else 0 for item in judgements], 
                device=log_probs.device,
                dtype=log_probs.dtype,
            )
            # cum_rewards = torch.cumsum(reward[1:].flip(0), dim=0).flip(0)
            baseline = torch.mean(reward)
            advantage = reward[1:] - baseline

            batch_loss += -torch.sum(advantage * log_probs[i, idxs])
            batch_reward += torch.sum(reward)

        return batch_reward / batch_size, batch_loss / batch_size
    
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

        batch_size = len(batch['questions'])
        batch_questions = batch['questions']
        batch_answers = batch['answers']
        batch_contexts = batch['contexts']

        # sparse retrive using BM25
        sparse_retrived_contexts = [
            bm25_sparse_retrieve(q, ctx, self.n_preselect)
            for q, ctx in zip(batch_questions, batch_contexts)
        ]
        # get retriver score 
        scores, probs, log_probs = self.retriver(batch_questions, sparse_retrived_contexts)
        # get final retrived contexts
        batch_relative_contexts = []
        for i in range(batch_size):
            k = min(self.n_shot, (probs[i] > 1e-6).sum().item())
            _, relative_context = self.select_context(sparse_retrived_contexts[i], probs[i], k)
            batch_relative_contexts.append(relative_context)

        for i in range(batch_size):
            contexts = batch_relative_contexts[i]
            generated_answer = self.base_model([contexts], batch_questions[i])

            judgement = self.evaluator(
                batch_questions[i] ,
                batch_answers[i], 
                generated_answer
            )

            acc = [1 if item == "NO" else 0 for item in judgement]
        
            results.extend(acc)
        
        return results
    
    def bm25_generate(self, batch):
        results = []

        batch_size = len(batch['questions'])
        batch_questions = batch['questions']
        batch_answers = batch['answers']
        batch_contexts = batch['contexts']

        retrived_contexts = [
            bm25_sparse_retrieve(q, ctx, self.n_shot)
            for q, ctx in zip(batch_questions, batch_contexts)
        ]

        for i in range(batch_size):
            contexts = retrived_contexts[i]
            generated_answer = self.base_model([contexts], batch_questions[i])

            judgement = self.evaluator(
                batch_questions[i] ,
                batch_answers[i], 
                generated_answer
            )

            acc = [1 if item == "NO" else 0 for item in judgement]
        
            results.extend(acc)
        
        return results
