#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Training with Active Learning Integration

This demonstrates how to integrate active learning with your existing
train_nodule_feature_cnn_model_v75.py training pipeline.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader

# Import your existing training components
from train_nodule_feature_cnn_model_v75 import (
    CONFIG,
    load_config,
    get_device,
    MultiTaskNoduleCNN_V60,
    NoduleFeatureDataset_V60,
    get_transforms_v60,
    train_epoch,
    validate_epoch,
    TrainingLogger
)

# Import active learning utilities
from active_learning_utils import ActiveLearningManager


def train_with_active_learning(
    config,
    initial_train_df,
    val_df,
    feature_mapping,
    image_root,
    num_al_cycles=3,
    samples_per_cycle=100,
    epochs_per_cycle=10
):
    """
    Training loop with active learning integration
    
    Args:
        config: Configuration dictionary
        initial_train_df: Initial training DataFrame
        val_df: Validation DataFrame
        feature_mapping: Feature mapping dictionary
        image_root: Root directory for images
        num_al_cycles: Number of active learning cycles
        samples_per_cycle: Samples to review per cycle
        epochs_per_cycle: Training epochs per cycle
    """
    
    # Setup
    device = get_device()
    output_dir = config['output_dir']
    al_output_dir = os.path.join(output_dir, 'active_learning')
    
    # Initialize logger
    logger = TrainingLogger(os.path.join(output_dir, 'training_with_al.log'))
    logger.log("🚀 Training with Active Learning")
    logger.log(f"   AL Cycles: {num_al_cycles}")
    logger.log(f"   Samples per cycle: {samples_per_cycle}")
    logger.log(f"   Epochs per cycle: {epochs_per_cycle}")
    
    # Initialize model
    model = MultiTaskNoduleCNN_V60(feature_mapping, dropout_rate=config['dropout_rate'])
    model = model.to(device)
    
    # Initialize active learning manager
    al_manager = ActiveLearningManager(
        model=model,
        config={
            'al_strategy': 'hybrid',  # or 'uncertainty', 'disagreement', 'diversity'
            'samples_per_cycle': samples_per_cycle
        },
        output_dir=al_output_dir
    )
    
    # Current training data (will be updated each cycle)
    current_train_df = initial_train_df.copy()
    
    # Active learning cycles
    for cycle in range(num_al_cycles):
        logger.log(f"\n{'='*80}")
        logger.log(f"Active Learning Cycle {cycle + 1}/{num_al_cycles}")
        logger.log(f"{'='*80}")
        logger.log(f"Current training samples: {len(current_train_df)}")
        
        # =====================================================================
        # PHASE 1: TRAIN MODEL
        # =====================================================================
        logger.log(f"\n📚 Phase 1: Training model for {epochs_per_cycle} epochs...")
        
        # Create datasets and loaders
        train_transform, val_transform = get_transforms_v60()
        train_dataset = NoduleFeatureDataset_V60(
            current_train_df, 
            image_root, 
            feature_mapping, 
            train_transform
        )
        val_dataset = NoduleFeatureDataset_V60(
            val_df, 
            image_root, 
            feature_mapping, 
            val_transform
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True,
            num_workers=config['num_workers']
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False,
            num_workers=config['num_workers']
        )
        
        # Training loop
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        
        best_val_f1 = 0
        for epoch in range(epochs_per_cycle):
            train_loss, train_metrics = train_epoch(
                model, train_loader, loss_fn, optimizer, device, config
            )
            val_loss, val_metrics = validate_epoch(
                model, val_loader, loss_fn, device
            )
            
            logger.log(f"  Epoch {epoch+1}/{epochs_per_cycle}: "
                      f"Train F1={train_metrics.get('bom_f1', 0):.4f}, "
                      f"Val F1={val_metrics.get('bom_f1', 0):.4f}")
            
            if val_metrics.get('bom_f1', 0) > best_val_f1:
                best_val_f1 = val_metrics.get('bom_f1', 0)
        
        logger.log(f"✅ Training complete. Best Val F1: {best_val_f1:.4f}")
        
        # =====================================================================
        # PHASE 2: IDENTIFY UNCERTAIN SAMPLES
        # =====================================================================
        logger.log(f"\n🔍 Phase 2: Identifying uncertain samples...")
        
        selected_indices, uncertainty_df = al_manager.identify_uncertain_samples(
            dataset=train_dataset,
            dataloader=train_loader,
            device=device,
            top_k=samples_per_cycle,
            extract_features=True  # For diversity sampling
        )
        
        # =====================================================================
        # PHASE 3: EXPORT FOR ANNOTATION
        # =====================================================================
        logger.log(f"\n📋 Phase 3: Exporting samples for annotation...")
        
        annotation_file = al_manager.export_for_annotation(
            selected_indices=selected_indices,
            uncertainty_df=uncertainty_df,
            dataset_df=current_train_df,
            image_root=image_root,
            cycle=cycle
        )
        
        # =====================================================================
        # PHASE 4: HUMAN ANNOTATION
        # =====================================================================
        logger.log(f"\n👤 Phase 4: Waiting for human annotations...")
        logger.log(f"\n   Please annotate the samples:")
        logger.log(f"   1. Run: python annotation_interface_simple.py {annotation_file}")
        logger.log(f"   2. Or manually edit: {annotation_file}")
        logger.log(f"   3. Save as: cycle_{cycle}_reviewed.csv")
        logger.log(f"   4. Press Enter to continue...")
        
        # Wait for user input
        input("\n   Press Enter after completing annotations...")
        
        # Import annotations
        try:
            annotations_df = al_manager.import_annotations(cycle)
        except FileNotFoundError:
            logger.log(f"   ⚠️  Annotation file not found. Skipping this cycle.")
            continue
        
        # =====================================================================
        # PHASE 5: UPDATE DATASET
        # =====================================================================
        logger.log(f"\n🔄 Phase 5: Updating dataset with corrections...")
        
        current_train_df, update_stats = al_manager.update_dataset(
            annotations_df=annotations_df,
            original_df=current_train_df
        )
        
        # Save updated dataset
        updated_dataset_path = os.path.join(
            al_output_dir,
            'results',
            f'updated_train_data_cycle_{cycle}.csv'
        )
        current_train_df.to_csv(updated_dataset_path, index=False)
        logger.log(f"   Updated dataset saved: {updated_dataset_path}")
        
        # =====================================================================
        # PHASE 6: EVALUATE IMPROVEMENT
        # =====================================================================
        logger.log(f"\n📊 Phase 6: Evaluating improvement...")
        
        # Retrain on updated data and evaluate
        # (This would be a full evaluation - simplified here)
        logger.log(f"   Labels corrected: {update_stats['num_corrected']}")
        logger.log(f"   Labels confirmed: {update_stats['num_confirmed']}")
        logger.log(f"   Samples flagged: {update_stats['num_flagged']}")
        
        # Save cycle results
        al_manager.save_cycle_results(
            cycle=cycle,
            metrics={
                'best_val_f1': best_val_f1,
                'train_samples': len(current_train_df),
                'corrections': update_stats
            }
        )
        
        logger.log(f"\n✅ Cycle {cycle + 1} complete!")
    
    # =========================================================================
    # FINAL TRAINING
    # =========================================================================
    logger.log(f"\n{'='*80}")
    logger.log(f"Final Training with All Corrections")
    logger.log(f"{'='*80}")
    logger.log(f"Final training samples: {len(current_train_df)}")
    
    # Train final model with all corrections
    # (Full training loop here - simplified)
    
    logger.log(f"\n🎉 Active Learning Training Complete!")
    logger.log(f"   Total AL cycles: {num_al_cycles}")
    logger.log(f"   Final dataset size: {len(current_train_df)}")
    
    return model, current_train_df


def main():
    """Main entry point"""
    
    # Load configuration
    config = load_config()
    
    # Update config for active learning
    config.update({
        'al_cycles': 3,
        'samples_per_cycle': 100,
        'epochs_per_cycle': 10,
        'al_strategy': 'hybrid'
    })
    
    # Load data (simplified - use your actual data loading)
    print("Loading data...")
    # initial_train_df = load_your_training_data()
    # val_df = load_your_validation_data()
    # feature_mapping = load_feature_mapping()
    # image_root = config['image_root']
    
    # For demonstration, we'll skip actual training
    print("\n" + "="*80)
    print("Active Learning Training Example")
    print("="*80)
    print("\nThis is a template. To use:")
    print("1. Uncomment data loading code above")
    print("2. Run: python train_with_active_learning_example.py")
    print("3. Follow the prompts for each AL cycle")
    print("\nKey features:")
    print("  ✅ Identifies uncertain samples automatically")
    print("  ✅ Exports for human review")
    print("  ✅ Updates dataset with corrections")
    print("  ✅ Retrains iteratively")
    print("  ✅ Tracks all changes")
    
    # Uncomment to run actual training:
    # model, final_train_df = train_with_active_learning(
    #     config=config,
    #     initial_train_df=initial_train_df,
    #     val_df=val_df,
    #     feature_mapping=feature_mapping,
    #     image_root=image_root,
    #     num_al_cycles=config['al_cycles'],
    #     samples_per_cycle=config['samples_per_cycle'],
    #     epochs_per_cycle=config['epochs_per_cycle']
    # )


if __name__ == '__main__':
    main()
