from model.attention import *

class BiLSTM(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128, dropout=0.3):
        super(BiLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = SelfAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict()

    def add_head(self, name, num_classes):
        self.heads[name] = nn.Linear(self.hidden_dim * 2, num_classes)
        return self

    def forward(self, x, head_names=None):
        x = x.float()
        if x.dim() == 2: x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        context_vector, attn_weights = self.attention(lstm_out)
        context_vector = self.dropout(context_vector)
        
        results = {'attention': attn_weights}
        target_heads = head_names if head_names else self.heads.keys()
        for name in target_heads:
            if name in self.heads:
                results[name] = self.heads[name](context_vector)
        return results
    @staticmethod
    def load_model(path, heads_config, device='cpu', input_dim=384, hidden_dim=128):
        model = BiLSTM(input_dim=input_dim, hidden_dim=hidden_dim)
        
        for name, num_classes in heads_config.items():
            model.add_head(name, num_classes)
            
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint.get('model_state', checkpoint)
        
        try:
            model.load_state_dict(state_dict, strict=True)
            model.to(device)
            model.eval()
            print(f"Successfully loaded weights from {path}")
        except RuntimeError as e:
            print(f"Error: {e}")
            
        return model

class MultiLayerDataset(Dataset):
    def __init__(self, embeddings, l1_multihot, l2_labels, l3_multihot):
        self.x = embeddings
        self.y = {
            'l1': torch.tensor(l1_multihot, dtype=torch.float),
            'l2': torch.tensor(l2_labels, dtype=torch.long),
            'l3': torch.tensor(l3_multihot, dtype=torch.float)
        }

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], {name: label[idx] for name, label in self.y.items()}

class MultiTaskLoss(nn.Module):
    def __init__(self, weights=None):
        super(MultiTaskLoss, self).__init__()
        self.weights = weights if weights else {'l1': 0.1, 'l2': 0.3, 'l3': 0.6}
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):
        total_loss = 0
        loss_dict = {}
        for name, pred in predictions.items():
            if name == 'attention' or name not in targets: continue
            loss = self.bce_loss(pred, targets[name].float())
            w = self.weights.get(name, 1.0)
            total_loss += w * loss
            loss_dict[f'loss_{name}'] = loss.item()
        return total_loss, loss_dict