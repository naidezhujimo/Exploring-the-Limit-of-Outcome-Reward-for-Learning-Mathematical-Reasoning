from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
from sympy import sympify, Eq, simplify
import random

class OREALModel(nn.Module):
    def __init__(self, base_model_name):
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            output_hidden_states=True,
            device_map='auto'
        )
        self.token_reward_layer = nn.Linear(self.base_model.config.hidden_size, 1)
        nn.init.zeros_(self.token_reward_layer.weight)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden = outputs.hidden_states[-1]  # [batch, seq_len, hidden]
        token_rewards = self.token_reward_layer(last_hidden).squeeze(-1)
        return outputs.logits, token_rewards
    
class MathDataset(Dataset):
    def __init__(self, problems, tokenizer, max_length=2048):
        self.problems = problems
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.problems)
    
    def __getitem__(self, idx):
        text = self.problems[idx]
        encoding = self.tokenizer(
            text,
            max_length = self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}

class OREALTrainer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = optim.AdamW([
            {'params': model.base_model.parameters(), 'lr': config['policy_lr']},
            {'params': model.token_reward_layer.parameters(), 'lr': config['reward_lr']}
        ])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        self.optimizer, 
        T_max=config['total_steps'],
        eta_min=config['policy_lr']/5
    )

        # 初始化参考模型
        self.ref_model = AutoModelForCausalLM.from_pretrained(config['base_model'])
        self.ref_model.requires_grad_(False)

    def compute_kl(self, logits, ref_logits, attention_mask):
        probs = F.log_softmax(logits, dim=-1)
        ref_probs = F.log_softmax(ref_logits, dim=-1)
        kl = (probs.exp() * (probs - ref_probs)).sum(dim=-1)
        return (kl * attention_mask).sum()/ attention_mask.sum()

    def train_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            # Best-of-N采样
            generations = []
            for _ in range(self.config['n_samples']):
                outputs = self.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_new_tokens=self.config['max_new_tokens'],
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7
                )
                generations.append(outputs)
            
            # 验证阶段
            rewards = [self.verifier(generation) for generation in generations]
            best_idx = np.argmax(rewards)
            best_generation = generations[best_idx]

        # 训练阶段
        self.model.train()
        self.optimizer.zero_grad()

        # 正样本处理
        pos_logits, pos_token_rewards = self.model(
            input_ids=best_generation['input_ids'],
            attention_mask=best_generation['attention_mask']
        )

        # 令牌级奖励计算
        omega_plus = torch.clamp(2 * torch.sigmoid(pos_token_rewards) - 1, min=0)
        pos_loss = - (omega_plus * F.log_softmax(pos_logits, dim=-1)).mean()
        
        # KL正则项
        with torch.no_grad():
            ref_logits = self.ref_model(
                input_ids=best_generation['input_ids'],
                attention_mask=best_generation['attention_mask']
            ).logits

        kl_loss = self.compute_kl(pos_logits, ref_logits, best_generation['attention_mask'])

        # 负样本处理
        neg_indices = [i for i, r in enumerate(rewards) if r == 0]
        neg_generation = generations[random.choice(neg_indices)] if neg_indices else None

        if neg_generation:
            neg_logits, neg_token_rewards = self.model(input_ids=neg_generation['input_ids'],
                                                 attention_mask=neg_generation['attention_mask'])
            omega_minus = torch.clamp(1 - 2 * torch.sigmoid(neg_token_rewards), min=0)
            
            ratio = (F.log_softmax(neg_logits, dim=-1) - 
                    F.log_softmax(self.ref_model(input_ids=neg_generation['input_ids'],
                                                 attention_mask=neg_generation['attention_mask']), dim=-1))
            neg_loss = (omega_minus * ratio).mean()

        # 总损失
        total_loss = pos_loss + self.config['kl_coef'] * kl_loss + self.config['neg_coef'] * neg_loss

        # 反向传播
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return {
            'loss': total_loss.item(),
            'pos_loss': pos_loss.item(),
            'neg_loss': neg_loss.item(),
            'kl_loss': kl_loss.item()
        }

config = {
    'base_model': "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    'n_samples': 16,
    'max_new_tokens': 1024,
    'policy_lr': 5e-7,
    'reward_lr': 2e-6,
    'kl_coef': 0.01,
    'neg_coef': 0.5,
    'batch_size': 4
}

# 初始化组件
model = OREALModel(config['base_model'])
tokenizer = AutoTokenizer.from_pretrained(config['base_model'])

math_problems = [
    "Solve for x: 2x + 5 = 15. Show your steps.",
    "Calculate the area of a circle with radius 3cm."
]

# 训练流程
dataset = MathDataset(math_problems, tokenizer)
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
trainer = OREALTrainer(model, tokenizer, config)

for epoch in range(10):
    for batch in dataloader:
        metrics = trainer.train_step(batch)
        print(f"Epoch {epoch} | Loss: {metrics['loss']:.4f} | Pos: {metrics['pos_loss']:.4f} | Neg: {metrics['neg_loss']:.4f} | KL: {metrics['kl_loss']:.4f}")

# 保存模型
model.save_pretrained("./oreal_trained")