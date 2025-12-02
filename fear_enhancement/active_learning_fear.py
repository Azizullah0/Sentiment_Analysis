# active_learning_fear.py
import numpy as np
from sklearn.cluster import KMeans

class FearActiveLearner:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def find_uncertain_fear_predictions(self, unlabeled_texts, batch_size=32):
        """Find texts where fear prediction is uncertain"""
        uncertainties = []
        
        for i in range(0, len(unlabeled_texts), batch_size):
            batch = unlabeled_texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, 
                max_length=512, return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                
                # Fear class is index 7
                fear_probs = probs[:, 7].cpu().numpy()
                
                # Uncertainty: close to decision boundary (0.5)
                uncertainty = 1 - np.abs(fear_probs - 0.5) * 2
                
                for j, (text, fear_prob, uncert) in enumerate(zip(batch, fear_probs, uncertainty)):
                    if 0.3 < fear_prob < 0.7:  # Uncertain region
                        uncertainties.append({
                            'text': text,
                            'fear_probability': float(fear_prob),
                            'uncertainty': float(uncert),
                            'all_probs': probs[j].cpu().numpy()
                        })
        
        # Sort by uncertainty
        uncertainties.sort(key=lambda x: x['uncertainty'], reverse=True)
        return uncertainties[:100]  # Top 100 most uncertain
    
    def select_diverse_fear_samples(self, uncertain_samples, n_samples=20):
        """Select diverse samples for labeling"""
        # Extract embeddings
        texts = [s['text'] for s in uncertain_samples]
        embeddings = self._get_embeddings(texts)
        
        # Cluster to ensure diversity
        kmeans = KMeans(n_clusters=n_samples, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Select one sample per cluster
        selected_indices = []
        for cluster_id in range(n_samples):
            cluster_samples = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_samples) > 0:
                # Select most uncertain in cluster
                cluster_uncertainties = [uncertain_samples[i]['uncertainty'] 
                                        for i in cluster_samples]
                most_uncertain_idx = cluster_samples[np.argmax(cluster_uncertainties)]
                selected_indices.append(most_uncertain_idx)
        
        return [uncertain_samples[i] for i in selected_indices]
    
    def _get_embeddings(self, texts):
        """Get sentence embeddings"""
        inputs = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.bert(**inputs)
            # Use [CLS] token embeddings
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings