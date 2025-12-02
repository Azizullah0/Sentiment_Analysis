# fear_enhanced_model.py
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class FearEnhancedEmotionClassifier(nn.Module):
    def __init__(self, base_model_name, num_emotions=8):
        super().__init__()
        
        # Base ParsBERT model
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=num_emotions
        )
        
        # Specialized fear detection head
        self.fear_detector = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Fear probability
            nn.Sigmoid()
        )
        
        # Fear-specific features extractor
        self.fear_keyword_encoder = self._create_fear_keyword_encoder()
        
    def _create_fear_keyword_encoder(self):
        """Create embeddings for fear-related keywords"""
        fear_keywords = [
            "می‌ترسم", "نگران", "هراس", "وحشت", "دلهره",
            "اضطراب", "ترس", "لرزش", "هراسان", "مضطرب"
        ]
        
        # This would use pre-trained embeddings
        # For now, return a simple lookup
        return nn.Embedding(len(fear_keywords), 32)
    
    def forward(self, input_ids, attention_mask, keyword_indices=None):
        # Base model predictions
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        base_logits = outputs.logits
        
        # Get hidden states for fear analysis
        hidden_states = self.base_model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        
        # [CLS] token representation
        cls_representation = hidden_states[:, 0, :]
        
        # Fear-specific prediction
        fear_prob = self.fear_detector(cls_representation)
        
        # Adjust base predictions based on fear probability
        fear_idx = 7  # Assuming fear is index 7
        fear_boost = fear_prob * 2.0  # Boost fear probability
        
        # Create adjusted logits
        adjusted_logits = base_logits.clone()
        adjusted_logits[:, fear_idx] += fear_boost.squeeze()
        
        return {
            'logits': adjusted_logits,
            'fear_probability': fear_prob,
            'base_logits': base_logits
        }

# Training with focal loss for fear
class FearFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, fear_idx=7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.fear_idx = fear_idx
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, logits, targets):
        # Standard cross entropy
        ce_loss = self.ce_loss(logits, targets)
        
        # Focal loss component for fear
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        # Extra weight for fear class
        fear_mask = (targets == self.fear_idx).float()
        fear_weight = 1.0 + self.alpha * fear_mask
        
        loss = (focal_weight * ce_loss * fear_weight).mean()
        return loss