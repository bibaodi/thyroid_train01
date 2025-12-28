#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Annotation Interface for Active Learning

This provides a basic command-line interface for reviewing uncertain samples.
For a web-based interface, see annotation_interface_web.py (requires streamlit)
"""

import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime


class SimpleAnnotationInterface:
    """Command-line interface for sample annotation"""
    
    def __init__(self, annotation_file: str):
        """
        Args:
            annotation_file: Path to CSV file with samples to annotate
        """
        self.annotation_file = annotation_file
        self.df = pd.read_csv(annotation_file)
        self.current_idx = 0
        self.annotations = []
        
        # Find where to resume (if partially annotated)
        if 'annotation_action' in self.df.columns:
            filled = self.df['annotation_action'].notna() & (self.df['annotation_action'] != '')
            if filled.any():
                self.current_idx = filled.sum()
                print(f"Resuming from sample {self.current_idx + 1}/{len(self.df)}")
    
    def display_sample(self, idx: int):
        """Display sample information and image"""
        row = self.df.iloc[idx]
        
        print("\n" + "="*80)
        print(f"Sample {idx + 1}/{len(self.df)}")
        print("="*80)
        
        # Display metadata
        print(f"\n📋 Sample Information:")
        print(f"   Access No: {row['access_no']}")
        print(f"   SOP UID: {row['sop_uid']}")
        print(f"   Current Label: {row['ground_truth']} ({'Benign' if row['ground_truth'] == 0 else 'Malignant'})")
        
        # Display model predictions
        print(f"\n🤖 Model Prediction:")
        print(f"   Predicted: {row['predicted_class']} ({'Benign' if row['predicted_class'] == 0 else 'Malignant'})")
        print(f"   Confidence: {row['max_prob']:.1%}")
        print(f"   P(True Label): {row['p_true']:.1%}")
        
        # Display uncertainty metrics
        print(f"\n📊 Uncertainty Metrics:")
        print(f"   Entropy: {row['entropy']:.3f}")
        print(f"   Margin: {row['margin']:.3f}")
        print(f"   Disagreement: {'Yes' if row['disagreement'] else 'No'}")
        
        # Display image if available
        if 'image_path' in row and pd.notna(row['image_path']):
            image_path = row['image_path']
            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    plt.figure(figsize=(8, 8))
                    plt.imshow(img)
                    plt.title(f"Current: {row['ground_truth']}, Predicted: {row['predicted_class']}")
                    plt.axis('off')
                    plt.show(block=False)
                    plt.pause(0.1)
                except Exception as e:
                    print(f"   ⚠️  Could not display image: {e}")
            else:
                print(f"   ⚠️  Image not found: {image_path}")
    
    def get_annotation(self) -> dict:
        """Get annotation from user"""
        print("\n" + "-"*80)
        print("What action do you want to take?")
        print("  1. Confirm current label is correct")
        print("  2. Correct label to Benign (0)")
        print("  3. Correct label to Malignant (1)")
        print("  4. Flag for expert review")
        print("  5. Skip this sample")
        print("  q. Quit and save progress")
        print("-"*80)
        
        while True:
            choice = input("\nYour choice (1-5, q): ").strip().lower()
            
            if choice == 'q':
                return {'action': 'quit'}
            
            if choice == '1':
                return {
                    'action': 'confirm',
                    'corrected_label': None,
                    'notes': input("Notes (optional): ").strip()
                }
            
            elif choice == '2':
                return {
                    'action': 'correct',
                    'corrected_label': 0,
                    'notes': input("Reason for correction: ").strip()
                }
            
            elif choice == '3':
                return {
                    'action': 'correct',
                    'corrected_label': 1,
                    'notes': input("Reason for correction: ").strip()
                }
            
            elif choice == '4':
                return {
                    'action': 'flag',
                    'corrected_label': None,
                    'notes': input("Reason for flagging: ").strip()
                }
            
            elif choice == '5':
                return {
                    'action': 'skip',
                    'corrected_label': None,
                    'notes': input("Reason for skipping: ").strip()
                }
            
            else:
                print("Invalid choice. Please enter 1-5 or q.")
    
    def save_annotation(self, idx: int, annotation: dict):
        """Save annotation to DataFrame"""
        self.df.at[idx, 'annotation_action'] = annotation['action']
        self.df.at[idx, 'corrected_label'] = annotation['corrected_label'] if annotation['corrected_label'] is not None else ''
        self.df.at[idx, 'annotator_notes'] = annotation['notes']
        self.df.at[idx, 'annotation_time'] = datetime.now().isoformat()
    
    def save_progress(self):
        """Save current progress to file"""
        output_file = self.annotation_file.replace('_for_review.csv', '_reviewed.csv')
        self.df.to_csv(output_file, index=False)
        print(f"\n💾 Progress saved to: {output_file}")
        return output_file
    
    def run(self):
        """Run the annotation interface"""
        print("\n" + "="*80)
        print("Active Learning Annotation Interface")
        print("="*80)
        print(f"\nTotal samples to review: {len(self.df)}")
        print(f"Starting from sample: {self.current_idx + 1}")
        print("\nPress Ctrl+C at any time to save and quit")
        
        try:
            while self.current_idx < len(self.df):
                # Display sample
                self.display_sample(self.current_idx)
                
                # Get annotation
                annotation = self.get_annotation()
                
                # Check for quit
                if annotation['action'] == 'quit':
                    print("\n👋 Quitting...")
                    break
                
                # Save annotation
                self.save_annotation(self.current_idx, annotation)
                
                # Move to next
                self.current_idx += 1
                
                # Close any open plots
                plt.close('all')
            
            # Save final results
            output_file = self.save_progress()
            
            # Print summary
            print("\n" + "="*80)
            print("Annotation Complete!")
            print("="*80)
            print(f"\nTotal annotated: {self.current_idx}")
            print(f"Output file: {output_file}")
            
            # Show action summary
            if 'annotation_action' in self.df.columns:
                print("\nAction Summary:")
                action_counts = self.df['annotation_action'].value_counts()
                for action, count in action_counts.items():
                    if action and action != '':
                        print(f"  - {action}: {count}")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            self.save_progress()
            print("Progress saved. You can resume later.")
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            self.save_progress()
            print("Progress saved despite error.")


def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python annotation_interface_simple.py <annotation_file.csv>")
        print("\nExample:")
        print("  python annotation_interface_simple.py annotations/cycle_0_for_review.csv")
        sys.exit(1)
    
    annotation_file = sys.argv[1]
    
    if not os.path.exists(annotation_file):
        print(f"Error: File not found: {annotation_file}")
        sys.exit(1)
    
    interface = SimpleAnnotationInterface(annotation_file)
    interface.run()


if __name__ == '__main__':
    main()
