#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Active Learning Utilities for Thyroid Nodule Classification

This module provides tools for:
1. Uncertainty estimation
2. Sample selection strategies
3. Annotation management
4. Dataset updating with corrections
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import entropy
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime


class UncertaintyEstimator:
    """Calculate various uncertainty metrics for model predictions"""
    
    @staticmethod
    def least_confident(probs: np.ndarray) -> float:
        """
        Uncertainty = 1 - max(p)
        Higher score = more uncertain
        """
        return 1 - np.max(probs)
    
    @staticmethod
    def margin_sampling(probs: np.ndarray) -> float:
        """
        Uncertainty = 1 - (p1 - p2)
        where p1, p2 are top-2 probabilities
        """
        sorted_probs = np.sort(probs)[::-1]
        if len(sorted_probs) < 2:
            return 0.0
        margin = sorted_probs[0] - sorted_probs[1]
        return 1 - margin
    
    @staticmethod
    def entropy_based(probs: np.ndarray) -> float:
        """
        Uncertainty = -Σ p*log(p)
        Normalized by log(n_classes)
        """
        eps = 1e-10
        probs = np.clip(probs, eps, 1 - eps)
        ent = entropy(probs)
        # Normalize by max entropy
        max_ent = np.log(len(probs))
        return ent / max_ent if max_ent > 0 else 0.0
    
    @staticmethod
    def oof_style(probs: np.ndarray, ground_truth: int, predicted_class: int) -> float:
        """
        Your current OOF approach:
        Uncertainty = (1 - p_true) * disagreement
        """
        p_true = probs[ground_truth]
        disagreement = 1.0 if predicted_class != ground_truth else 0.0
        return (1 - p_true) * disagreement
    
    @staticmethod
    def calculate_all(probs: np.ndarray, ground_truth: Optional[int] = None) -> Dict[str, float]:
        """Calculate all uncertainty metrics"""
        predicted_class = np.argmax(probs)
        
        metrics = {
            'least_confident': UncertaintyEstimator.least_confident(probs),
            'margin': UncertaintyEstimator.margin_sampling(probs),
            'entropy': UncertaintyEstimator.entropy_based(probs),
            'predicted_class': int(predicted_class),
            'max_prob': float(np.max(probs))
        }
        
        if ground_truth is not None:
            metrics['oof_style'] = UncertaintyEstimator.oof_style(probs, ground_truth, predicted_class)
            metrics['p_true'] = float(probs[ground_truth])
            metrics['disagreement'] = int(predicted_class != ground_truth)
        
        return metrics



class QueryStrategy:
    """Sample selection strategies for active learning"""
    
    def __init__(self, strategy: str = 'hybrid'):
        """
        Args:
            strategy: 'uncertainty', 'disagreement', 'diversity', 'hybrid'
        """
        self.strategy = strategy
        
    def select_samples(
        self, 
        uncertainties: List[Dict],
        features: Optional[np.ndarray] = None,
        top_k: int = 100,
        diversity_weight: float = 0.2
    ) -> List[int]:
        """
        Select top-k samples for annotation
        
        Args:
            uncertainties: List of uncertainty dicts for each sample
            features: Feature vectors for diversity sampling (optional)
            top_k: Number of samples to select
            diversity_weight: Weight for diversity in hybrid strategy
            
        Returns:
            List of selected sample indices
        """
        if self.strategy == 'uncertainty':
            return self._uncertainty_sampling(uncertainties, top_k)
        elif self.strategy == 'disagreement':
            return self._disagreement_sampling(uncertainties, top_k)
        elif self.strategy == 'diversity':
            return self._diversity_sampling(uncertainties, features, top_k)
        elif self.strategy == 'hybrid':
            return self._hybrid_sampling(uncertainties, features, top_k, diversity_weight)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _uncertainty_sampling(self, uncertainties: List[Dict], top_k: int) -> List[int]:
        """Select samples with highest entropy"""
        scores = [(i, u['entropy']) for i, u in enumerate(uncertainties)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores[:top_k]]
    
    def _disagreement_sampling(self, uncertainties: List[Dict], top_k: int) -> List[int]:
        """Select samples where model disagrees with ground truth"""
        scores = [(i, u.get('oof_style', 0)) for i, u in enumerate(uncertainties)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores[:top_k]]
    
    def _diversity_sampling(
        self, 
        uncertainties: List[Dict], 
        features: np.ndarray, 
        top_k: int
    ) -> List[int]:
        """Select diverse samples using k-means clustering"""
        if features is None:
            return self._uncertainty_sampling(uncertainties, top_k)
        
        # First, select top-2*k uncertain samples
        uncertain_indices = self._uncertainty_sampling(uncertainties, min(top_k * 2, len(uncertainties)))
        uncertain_features = features[uncertain_indices]
        
        # Then cluster and select one from each cluster
        n_clusters = min(top_k, len(uncertain_indices))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(uncertain_features)
        
        # Select sample closest to each cluster center
        selected = []
        for cluster_id in range(n_clusters):
            cluster_mask = clusters == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            if len(cluster_indices) > 0:
                # Select most uncertain in this cluster
                cluster_uncertainties = [uncertainties[uncertain_indices[i]]['entropy'] 
                                       for i in cluster_indices]
                best_in_cluster = cluster_indices[np.argmax(cluster_uncertainties)]
                selected.append(uncertain_indices[best_in_cluster])
        
        return selected[:top_k]
    
    def _hybrid_sampling(
        self, 
        uncertainties: List[Dict], 
        features: Optional[np.ndarray], 
        top_k: int,
        diversity_weight: float
    ) -> List[int]:
        """Combine uncertainty and diversity"""
        # Calculate combined scores
        scores = []
        for i, u in enumerate(uncertainties):
            uncertainty_score = (
                0.4 * u['entropy'] + 
                0.3 * u.get('oof_style', 0) + 
                0.3 * u['margin']
            )
            scores.append((i, uncertainty_score))
        
        # Sort by uncertainty
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # If no features, return top-k by uncertainty
        if features is None or diversity_weight == 0:
            return [idx for idx, _ in scores[:top_k]]
        
        # Otherwise, add diversity
        selected = []
        selected_features = []
        
        for idx, score in scores:
            if len(selected) >= top_k:
                break
            
            # Calculate diversity bonus
            if len(selected_features) == 0:
                diversity_bonus = 0
            else:
                # Distance to nearest selected sample
                distances = np.linalg.norm(
                    features[idx] - np.array(selected_features), 
                    axis=1
                )
                diversity_bonus = np.min(distances)
            
            # Combined score
            final_score = score + diversity_weight * diversity_bonus
            
            selected.append(idx)
            selected_features.append(features[idx])
        
        return selected



class ActiveLearningManager:
    """Main class for managing active learning workflow"""
    
    def __init__(self, model, config: Dict, output_dir: str):
        """
        Args:
            model: Trained PyTorch model
            config: Configuration dictionary
            output_dir: Directory for saving annotations and results
        """
        self.model = model
        self.config = config
        self.output_dir = output_dir
        self.annotation_history = []
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
        
        # Initialize query strategy
        self.query_strategy = QueryStrategy(
            strategy=config.get('al_strategy', 'hybrid')
        )
        
    def identify_uncertain_samples(
        self,
        dataset,
        dataloader,
        device,
        top_k: int = 100,
        extract_features: bool = False
    ) -> Tuple[List[int], pd.DataFrame]:
        """
        Identify most uncertain samples for annotation
        
        Args:
            dataset: PyTorch dataset
            dataloader: DataLoader for the dataset
            device: torch device
            top_k: Number of samples to select
            extract_features: Whether to extract features for diversity sampling
            
        Returns:
            selected_indices: List of selected sample indices
            uncertainty_df: DataFrame with uncertainty metrics for all samples
        """
        self.model.eval()
        
        all_uncertainties = []
        all_features = [] if extract_features else None
        all_metadata = []
        
        print(f"\n🔍 Calculating uncertainties for {len(dataset)} samples...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move to device
                images = batch['image'].to(device)
                bom_labels = batch['bom'].cpu().numpy()
                
                # Get predictions
                outputs = self.model(images)
                bom_logits = outputs['bom']
                probs = F.softmax(bom_logits, dim=1).cpu().numpy()
                
                # Extract features if requested
                if extract_features:
                    # Get features from model backbone
                    features = self.model.backbone(images).cpu().numpy()
                    all_features.extend(features)
                
                # Calculate uncertainties
                for i in range(len(images)):
                    uncertainty_metrics = UncertaintyEstimator.calculate_all(
                        probs[i],
                        ground_truth=int(bom_labels[i])
                    )
                    
                    # Add metadata
                    sample_idx = batch_idx * dataloader.batch_size + i
                    uncertainty_metrics.update({
                        'sample_idx': sample_idx,
                        'access_no': batch['access_no'][i],
                        'sop_uid': batch['sop_uid'][i],
                        'ground_truth': int(bom_labels[i])
                    })
                    
                    all_uncertainties.append(uncertainty_metrics)
                    all_metadata.append({
                        'access_no': batch['access_no'][i],
                        'sop_uid': batch['sop_uid'][i]
                    })
        
        # Convert to DataFrame
        uncertainty_df = pd.DataFrame(all_uncertainties)
        
        # Select samples
        features_array = np.array(all_features) if extract_features else None
        selected_indices = self.query_strategy.select_samples(
            all_uncertainties,
            features=features_array,
            top_k=top_k
        )
        
        print(f"✅ Selected {len(selected_indices)} samples for annotation")
        print(f"   Strategy: {self.config.get('al_strategy', 'hybrid')}")
        print(f"   Avg uncertainty (entropy): {uncertainty_df.loc[selected_indices, 'entropy'].mean():.3f}")
        print(f"   Disagreement rate: {uncertainty_df.loc[selected_indices, 'disagreement'].mean():.1%}")
        
        return selected_indices, uncertainty_df

    
    def export_for_annotation(
        self,
        selected_indices: List[int],
        uncertainty_df: pd.DataFrame,
        dataset_df: pd.DataFrame,
        image_root: str,
        cycle: int
    ) -> str:
        """
        Export selected samples for human annotation
        
        Args:
            selected_indices: Indices of selected samples
            uncertainty_df: DataFrame with uncertainty metrics
            dataset_df: Original dataset DataFrame
            image_root: Root directory for images
            cycle: Active learning cycle number
            
        Returns:
            output_path: Path to exported CSV file
        """
        # Get selected samples
        selected_df = uncertainty_df.loc[selected_indices].copy()
        
        # Merge with original dataset to get all features
        selected_df = selected_df.merge(
            dataset_df,
            on=['access_no', 'sop_uid'],
            how='left',
            suffixes=('_uncertainty', '_original')
        )
        
        # Add image paths
        selected_df['image_path'] = selected_df.apply(
            lambda row: os.path.join(image_root, str(row['access_no']), f"{row['sop_uid']}.jpg"),
            axis=1
        )
        
        # Add annotation columns
        selected_df['annotation_action'] = ''  # To be filled by annotator
        selected_df['corrected_label'] = ''
        selected_df['annotator_notes'] = ''
        selected_df['annotation_time'] = ''
        selected_df['cycle'] = cycle
        
        # Save
        output_path = os.path.join(
            self.output_dir, 
            'annotations', 
            f'cycle_{cycle}_for_review.csv'
        )
        selected_df.to_csv(output_path, index=False)
        
        print(f"\n📋 Exported {len(selected_df)} samples to: {output_path}")
        print(f"\n   Annotation Instructions:")
        print(f"   1. Open: {output_path}")
        print(f"   2. For each sample, fill in:")
        print(f"      - annotation_action: 'confirm', 'correct', 'flag', or 'skip'")
        print(f"      - corrected_label: 0 (benign) or 1 (malignant) if correcting")
        print(f"      - annotator_notes: Any observations")
        print(f"   3. Save as: cycle_{cycle}_reviewed.csv")
        print(f"   4. Run: python import_annotations.py {cycle}")
        
        return output_path
    
    def import_annotations(self, cycle: int) -> pd.DataFrame:
        """
        Import human annotations from CSV
        
        Args:
            cycle: Active learning cycle number
            
        Returns:
            annotations_df: DataFrame with annotations
        """
        input_path = os.path.join(
            self.output_dir,
            'annotations',
            f'cycle_{cycle}_reviewed.csv'
        )
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Annotation file not found: {input_path}")
        
        annotations_df = pd.read_csv(input_path)
        
        # Validate
        required_cols = ['annotation_action', 'access_no', 'sop_uid']
        missing_cols = [col for col in required_cols if col not in annotations_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter valid annotations
        valid_actions = ['confirm', 'correct', 'flag', 'skip']
        annotations_df = annotations_df[
            annotations_df['annotation_action'].isin(valid_actions)
        ]
        
        print(f"\n✅ Imported {len(annotations_df)} annotations from cycle {cycle}")
        print(f"   Actions:")
        for action in valid_actions:
            count = (annotations_df['annotation_action'] == action).sum()
            print(f"   - {action}: {count}")
        
        # Save to history
        self.annotation_history.append({
            'cycle': cycle,
            'timestamp': datetime.now().isoformat(),
            'num_annotations': len(annotations_df),
            'actions': annotations_df['annotation_action'].value_counts().to_dict()
        })
        
        return annotations_df

    
    def update_dataset(
        self,
        annotations_df: pd.DataFrame,
        original_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Update dataset with corrected labels
        
        Args:
            annotations_df: DataFrame with annotations
            original_df: Original dataset DataFrame
            
        Returns:
            updated_df: Updated dataset
            stats: Statistics about updates
        """
        updated_df = original_df.copy()
        
        stats = {
            'num_corrected': 0,
            'num_confirmed': 0,
            'num_flagged': 0,
            'corrections': []
        }
        
        for _, row in annotations_df.iterrows():
            action = row['annotation_action']
            
            # Find matching row in original dataset
            mask = (
                (updated_df['access_no'] == row['access_no']) &
                (updated_df['sop_uid'] == row['sop_uid'])
            )
            
            if not mask.any():
                continue
            
            if action == 'correct':
                # Update label
                old_label = updated_df.loc[mask, 'bom'].values[0]
                new_label = int(row['corrected_label'])
                updated_df.loc[mask, 'bom'] = new_label
                
                stats['num_corrected'] += 1
                stats['corrections'].append({
                    'access_no': row['access_no'],
                    'sop_uid': row['sop_uid'],
                    'old_label': int(old_label),
                    'new_label': new_label,
                    'notes': row.get('annotator_notes', '')
                })
                
            elif action == 'confirm':
                stats['num_confirmed'] += 1
                
            elif action == 'flag':
                # Mark for expert review
                updated_df.loc[mask, 'flagged_for_expert'] = 1
                stats['num_flagged'] += 1
        
        print(f"\n✅ Dataset updated:")
        print(f"   - Labels corrected: {stats['num_corrected']}")
        print(f"   - Labels confirmed: {stats['num_confirmed']}")
        print(f"   - Flagged for expert: {stats['num_flagged']}")
        
        # Save correction log
        if stats['corrections']:
            log_path = os.path.join(
                self.output_dir,
                'results',
                f'corrections_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
            with open(log_path, 'w') as f:
                json.dump(stats['corrections'], f, indent=2)
            print(f"   - Correction log saved: {log_path}")
        
        return updated_df, stats
    
    def save_cycle_results(self, cycle: int, metrics: Dict):
        """Save results for this active learning cycle"""
        results_path = os.path.join(
            self.output_dir,
            'results',
            f'cycle_{cycle}_results.json'
        )
        
        results = {
            'cycle': cycle,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'annotation_history': self.annotation_history
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"   - Results saved: {results_path}")
