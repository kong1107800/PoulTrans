import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(EnhancedLoss, self).__init__()
        self.margin = margin
        self.linear_layer = nn.Linear(2048, 1198)
        # 可学习的损失权重
        self.image_loss_weight = nn.Parameter(torch.tensor(0.5))
        self.text_loss_weight = nn.Parameter(torch.tensor(0.5))
    def forward(self, image_output, text_output, text_target):
        image_mapped = self.linear_layer(image_output.reshape(-1, 2048))
        selected_indices = torch.randperm(image_mapped.size(0))[:108].to("cuda:1")
        image_mapped = torch.index_select(image_mapped, 0, selected_indices)
        image_loss = self.contrastive_loss(image_mapped, text_output, self.margin)
        text_loss = F.cross_entropy(text_output, text_target)
        # 使用可学习的系数调整损失
        total_loss = self.image_loss_weight * image_loss + self.text_loss_weight * text_loss
        return total_loss

    def contrastive_loss(self, output, target, margin):
        #print(output.unsqueeze(1).shape)
        #print(target.unsqueeze(0).shape)
        similarity_matrix = F.cosine_similarity(output.unsqueeze(1), target.unsqueeze(0), dim=2)
        #positive_pairs_loss = torch.diagonal(similarity_matrix, offset=0, dim1=0, dim2=1)
        positive_pairs_loss = 1 - torch.diagonal(similarity_matrix, offset=0, dim1=0, dim2=1)
        negative_pairs_loss = torch.max(similarity_matrix - margin, torch.zeros_like(similarity_matrix))
        negative_pairs_loss = torch.sum(negative_pairs_loss, dim=1) - margin
        negative_pairs_loss = negative_pairs_loss[:len(positive_pairs_loss)]
        
        loss = torch.mean(positive_pairs_loss + negative_pairs_loss)
        return loss
