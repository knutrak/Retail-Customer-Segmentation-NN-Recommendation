import torch
from torch import nn
class NCF(nn.Module):
    def __init__(self, num_users, num_items, num_clusters, embedding_dim=64, cluster_embedding_dim=8):
        super(NCF, self).__init__()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.cluster_embedding = nn.Embedding(num_clusters, cluster_embedding_dim)
        
        # Dense layers
        # input size = user_embedding + item_embedding + cluster_embedding
        self.fc1 = nn.Linear(embedding_dim * 2 + cluster_embedding_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, user, item, cluster):
        # Get embeddings
        user_vec = self.user_embedding(user)
        item_vec = self.item_embedding(item)
        cluster_vec = self.cluster_embedding(cluster)
        
        # Concatenate
        x = torch.cat([user_vec, item_vec, cluster_vec], dim=1)
        
        # Dense layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))
        
        # Output
        return self.sigmoid(self.output(x))