import torch
import torch.nn as nn
import torch.nn.functional as F

class IdentityEmbeddingModule(nn.Module):
    """
    如果你本身就有 (B,T,embed_dim) 的输入, 这个模块只需要传回原值就行.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.output_dim = embed_dim  # 关键: 让外部知道embedding维度
    def forward(self, x):
        # x 期望形状: (B, T, embed_dim)
        # 直接原样返回即可
        return x

class SequenceRewardModel(nn.Module):
    """
    - embedding_module: 接受 (B,T,embed_dim) => (B,T,embedding_dim2), 也可以是 Identity
    - use_rnn=True => 用GRU对embedding序列做时序处理, 产出 step-wise reward
    - use_rnn=False => 直接对embedding做MLP (每步独立)
    """
    def __init__(self, embedding_module, hidden_size=128, use_rnn=True):
        super().__init__()
        self.embedding_module = embedding_module
        self.use_rnn = use_rnn
        self.hidden_size = hidden_size

        # 假设embedding_module输出: (B,T, embed_dim_out)
        embed_dim_out = self.embedding_module.output_dim

        if use_rnn:
            self.rnn = nn.GRU(embed_dim_out, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)  # step-wise
        else:
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim_out, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )

    def forward(self, emb_seq):
        """
        emb_seq: (B, T, E_in)  -- 先过embedding_module => (B, T, E_out)
        return: (B, T) step-wise reward
        """
        # 1) 做embedding, 如果embedding_module是Identity, 就原样返回
        x = self.embedding_module(emb_seq)  
        # shape (B, T, E_out)

        if self.use_rnn:
            out, _ = self.rnn(x)                # (B, T, hidden_size)
            rewards = self.fc(out).squeeze(-1)  # (B, T)
        else:
            B, T, E = x.shape
            x2 = x.reshape(B*T, E)
            out = self.mlp(x2)                  # (B*T,1)
            rewards = out.view(B, T)
        return rewards

    @torch.no_grad()
    def predict_stepwise(self, single_emb_seq):
        """
        single_emb_seq: (T, E_in)
        return (T,) step-wise
        """
        self.eval()
        x = single_emb_seq.unsqueeze(0)       # => (1,T,E_in)
        rew_seq = self.forward(x)             # => (1,T)
        return rew_seq.squeeze(0)             # => (T,)



class SequenceRewardTrainer:
    def __init__(self, model, lr=1e-4, device='cpu'):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_on_batch(self, emb_seq_batch, final_rewards_batch):
        """
        :param emb_seq_batch: (B, T, E_in), 统一长度
        :param final_rewards_batch: (B,)
        """
        self.model.train()
        emb_seq_batch = emb_seq_batch.to(self.device)
        final_rewards_batch = final_rewards_batch.to(self.device)

        pred_step_rewards = self.model(emb_seq_batch)  # (B, T)
        pred_final = pred_step_rewards.sum(dim=1)       # (B,)

        loss = F.mse_loss(pred_final, final_rewards_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict_stepwise(self, single_emb_seq):
        """
        single_emb_seq: (T, E_in)
        return: (T,) step-wise
        """
        return self.model.predict_stepwise(single_emb_seq.to(self.device))



if __name__=="__main__":
    # 例: 你外部已经把一个视频(原本 100 帧, c=xxx,h=xxx,w=xxx) 压缩成 (T=10, embed_dim=64)
    # B=2 条轨迹 => shape (2,10,64)
    B,T,E_in = 2, 10, 64
    emb_seq_batch = torch.randn(B,T,E_in)
    final_rewards_batch = torch.tensor([1.0, 0.0])

    # 构造embedding_module: 如果你已经embedding好, 直接用 Identity
    # 当然, 你也可以自己再加一些transform,project之类
    embedding_module = IdentityEmbeddingModule(embed_dim=E_in)

    # 构造RewardModel: use_rnn=True => 用GRU, hidden_size=128
    reward_model = SequenceRewardModel(embedding_module, hidden_size=128, use_rnn=True)
    trainer = SequenceRewardTrainer(reward_model, lr=1e-4, device='cpu')

    # 训练一批
    loss_val = trainer.train_on_batch(emb_seq_batch, final_rewards_batch)
    print("train_on_batch => loss=", loss_val)

    # 预测
    single_emb_seq = torch.randn(T,E_in)
    stepwise = trainer.predict_stepwise(single_emb_seq)
    print("stepwise=", stepwise)
    print("sum=", stepwise.sum().item())
