import os
import sys
import torch
from torchvision import transforms
from PIL import Image
import json
import argparse
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# Add the project root to Python path for proper imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from srcs.lib.nn.MultiTaskNoduleCNN import MultiTaskNoduleCNN
from srcs.lib.image_process.image_transforms import get_transforms_keep_aspect_ratio, get_transforms_center_pad_only
from srcs.lib.utils.time_stats_collector import TimeStatsCollector
from srcs.lib.filters.case_data_filter import CaseDataFilter, create_default_filter_config

# Configuration parameters
CONFIG = {
    'model_path0': '/home/eton/job-trainThyUS/59-result-models/model-efficientnet-eton251024/nodule_feature_cnn_v75/nodule_feature_cnn_v75_best_auc.pth',
    'model_path1':'/home/eton/job-trainThyUS/71-datasets/251016-efficientNetDatas/models/nodule_feature_cnn_v75/nodule_feature_cnn_v75_best_auc.pth',
    'model_path':'/home/eton/job-trainThyUS/59-result-models/efficientnet-models-start251024/thyUSCls-v251102F1.7/nodule_feature_cnn_v76_best_auc.pth',
    'feature_mapping_file': '/home/eton/job-trainThyUS/71-datasets/251016-efficientNetDatas/dataset/all_features_mapping_numer_v4.json',
    'threshold': 0.5,
    'malignant':1,'benign':0,
    'input_image_size': (224, 224),
    'imageNet_normalized_mean': [0.485, 0.456, 0.406],
    'imageNet_normalized_std': [0.229, 0.224, 0.225],
    'process_correct_predictions': False
}
demo_use='(google_ai_edgePy311) eton@M2703:~/job-trainThyUS/71-datasets/251016-efficientNetDatas/srcs$ python predict.py /tmp/input02/ --spreadsheet_file ../dataset/all_verify_sop_with_predictions.csv --output_folder /tmp/o02.1'
print(f"Demo use: {demo_use}")

__DEBUG__ = 0

# MultiTaskNoduleCNN has been moved to `srcs/models/MultiTaskNoduleCNN.py`

class Predictor:
    def __init__(self, model_path=None, feature_mapping_file=None, threshold=None, spreadsheet_file=None, process_correct_predictions=None, filter_config=None, orientation_model_path=None, thyroid_nodule_model_path=None):
        self.m_model_path = model_path or CONFIG['model_path']
        self.m_feature_mapping_file = feature_mapping_file or CONFIG['feature_mapping_file']
        self.m_threshold = threshold or CONFIG['threshold']
        self.m_spreadsheet_file = spreadsheet_file
        self.m_process_correct_predictions = process_correct_predictions if process_correct_predictions is not None else CONFIG['process_correct_predictions']
        self.m_orientation_model_path = orientation_model_path
        self.m_thyroid_nodule_model_path = thyroid_nodule_model_path
        self.m_config = {}
        self.m_config['image_normalized_mean'] = CONFIG['imageNet_normalized_mean']
        self.m_config['image_normalized_std'] = CONFIG['imageNet_normalized_std']
        self.m_time_stats = TimeStatsCollector()

        # Initialize case data filter
        self._init_data_filter(filter_config)
        self._init_basics()

        self.m_ground_truth_data = None
        if self.m_spreadsheet_file:
            self._load_ground_truth_data()

        img_size = CONFIG['input_image_size']
        _, self.m_transform = get_transforms_keep_aspect_ratio(img_size)
        #_, self.m_transform = get_transforms_center_pad_only()

    def _init_basics(self):
        # Get device
        self.m_device = self._get_device()
        print(f"Using device: {self.m_device}")
        print(f"Process correct predictions: {self.m_process_correct_predictions}")
        self.m_feature_mapping = self._load_feature_mapping()
        self.m_model = self._create_and_load_model(self.m_model_path, self.m_feature_mapping, self.m_device)


    def _init_data_filter(self, filter_config):
        self.m_filter_config = filter_config or create_default_filter_config()
        self.m_caseDataChecker = CaseDataFilter(self.m_filter_config, self.m_orientation_model_path, self.m_thyroid_nodule_model_path)
        print(f"Case filter initialized with nodes: {self.m_caseDataChecker.get_available_filters()}")

    def _get_device(self):
        if torch.backends.mps.is_available(): return torch.device("mps")
        if torch.cuda.is_available(): return torch.device("cuda")
        return torch.device("cpu")


    def _load_feature_mapping(self):
        """Load feature mapping file"""
        if not os.path.exists(self.m_feature_mapping_file):
            print(f"Error: Feature mapping file does not exist: {self.m_feature_mapping_file}")
            return None

        with open(self.m_feature_mapping_file, 'r', encoding='utf-8') as f:
            feature_mapping = json.load(f)

        return feature_mapping

    def _load_ground_truth_data(self):
        """Load ground truth data from spreadsheet file"""
        if not os.path.exists(self.m_spreadsheet_file):
            print(f"Error: Spreadsheet file does not exist: {self.m_spreadsheet_file}")
            self.m_ground_truth_data = None
            return

        try:
            # Select reading method based on file extension
            file_ext = os.path.splitext(self.m_spreadsheet_file)[1].lower()
            if file_ext == '.csv':
                df = pd.read_csv(self.m_spreadsheet_file)
            elif file_ext in ['.xls', '.xlsx']:
                df = pd.read_excel(self.m_spreadsheet_file)
            elif file_ext == '.ods':
                df = pd.read_excel(self.m_spreadsheet_file, engine='odf')
            else:
                print(f"Error: Unsupported file format: {file_ext}")
                self.m_ground_truth_data = None
                return

            # Check if required columns exist
            required_columns = ['access_no', 'sop_uid', 'bom']
            if not all(col in df.columns for col in required_columns):
                print(f"Error: Spreadsheet missing required columns. Need: {required_columns}")
                self.m_ground_truth_data = None
                return

            # Keep only required columns
            df = df[required_columns]

            # Convert access_no to string to ensure consistent type during matching
            df['access_no'] = df['access_no'].astype(str)
            df['sop_uid'] = df['sop_uid'].astype(str)

            # Create data structure for quick lookup
            self.m_ground_truth_data = {(row['access_no'], row['sop_uid']): row['bom']
                                     for _, row in df.iterrows()}
            print(f"Successfully loaded ground truth data, total {len(self.m_ground_truth_data)} records")
        except Exception as e:
            print(f"Error loading spreadsheet data: {str(e)}")
            self.m_ground_truth_data = None

    def _get_ground_truth(self, access_no, sop_uid):
        """Get ground truth label based on access_no and sop_uid"""
        if self.m_ground_truth_data is None:
            return None

        # Convert to strings to ensure matching
        access_no_str = str(access_no)
        sop_uid_str = str(sop_uid)

        return self.m_ground_truth_data.get((access_no_str, sop_uid_str), None)

    def _create_and_load_model(self, model_path, feature_mapping_file, compute_dev):
        """Create model instance and load pre-trained weights, with TorchScript optimization"""
        if not os.path.exists(model_path):
            print(f"Error: Model file does not exist: {model_path}")
            return None

        if self.feature_mapping is None:
            print("Error: Feature mapping not loaded, cannot create model")
            return None

        model = MultiTaskNoduleCNN(feature_mapping_file, dropout_rate=0.4).to(compute_dev)
        model.load_state_dict(torch.load(model_path, map_location=compute_dev))

        # Convert to TorchScript for improved inference speed
        model.eval()
        dummy_input = torch.randn(1, 3, CONFIG['input_image_size'][0], CONFIG['input_image_size'][1]).to(compute_dev)
        try:
            model = torch.jit.trace(model, dummy_input)
            model = torch.jit.freeze(model)
            print("Model optimized with TorchScript")
        except Exception as e:
            print(f"TorchScript optimization failed: {e}")

        print("Model loaded successfully")

        return model

    def _get_image_transform(self, img_size=(224,224)):
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.m_config['image_normalized_mean'],
                std=self.m_config['image_normalized_std']
            )
        ])
        return transform

    def _predict_batch_images(self, image_paths, batch_size=16):
        """Predict images in batches"""
        # Record prediction start time
        pred_start_time = self.m_time_stats.start_prediction_timer()

        # Initialize results list
        results = []

        # Process images in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            batch_tensors = []

            # Preprocess image batch
            for img_path in batch_paths:
                image = Image.open(img_path).convert('RGB')
                image_tensor_tsfmed = self.m_transform(image).unsqueeze(0)  # Add batch dimension
                image_tensor = image_tensor_tsfmed.to(self.m_device)
                batch_images.append((image_tensor_tsfmed, img_path))
                batch_tensors.append(image_tensor)
            # Create batch tensor
            batch_tensor = torch.cat(batch_tensors)

            # Batch inference
            self.m_model.eval()
            with torch.no_grad():
                outputs = self.m_model(batch_tensor)
                bom_logits = outputs['bom']
                bom_softmax = torch.softmax(bom_logits, dim=1)

                # Process results for each image
                for j, (tensor_tsfmed, img_path) in enumerate(batch_images):
                    benignProbs = bom_softmax[j, 0].item()
                    MalignProbs = bom_softmax[j, 1].item()

                    if MalignProbs > benignProbs:
                        prediction = CONFIG['malignant']
                        bom_probs = MalignProbs
                    else:
                        prediction = CONFIG['benign']
                        bom_probs = benignProbs

                    pred_idx = torch.argmax(bom_logits[j], dim=0)
                    if pred_idx.item() != prediction:
                        print(f"Warning: Prediction mismatch for {img_path}: pred_idx={pred_idx.item()}, computed_prediction={prediction}")

                    results.append({
                        'image_path': img_path,
                        'prediction': prediction,
                        'confidence': bom_probs,
                        'label': 'Malignant' if prediction == 1 else 'Benign',
                        'image_tensor_tsfmed': tensor_tsfmed
                    })

        # End prediction time recording
        self.m_time_stats.end_prediction_timer(pred_start_time)

        return results

    def _predict_single_image(self, image_path):
        """Predict a single image"""
        if self.m_model is None:
            print("Error: Model not loaded, cannot perform prediction")
            return None

        # Open and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor_tsfmed = self.m_transform(image).unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor_tsfmed.to(self.m_device)

        # Inference with time tracking
        pred_start_time = self.m_time_stats.start_prediction_timer()
        self.m_model.eval()
        with torch.no_grad():
            outputs = self.m_model(image_tensor)
            bom_logits = outputs['bom']
            bom_softmax = torch.softmax(bom_logits, dim=1)
            benignProbs = bom_softmax[0, 0].item()  # Probability of class 0
            MalignProbs = bom_softmax[0, 1].item()  # Probability of class 1
            if MalignProbs > benignProbs:
                prediction = CONFIG['malignant']
                bom_probs = MalignProbs
            else:
                prediction = CONFIG['benign']
                bom_probs = benignProbs
            pred_idx = torch.argmax(bom_logits, dim=1)
            if pred_idx.item() != prediction:
                print(f"Warning: Prediction mismatch for {image_path}: pred_idx={pred_idx.item()}, computed_prediction={prediction}")
        self.m_time_stats.end_prediction_timer(pred_start_time)

        # Return both the result and the transformed image tensor for visualization
        return {
            'image_path': image_path,
            'prediction': prediction,
            'confidence': bom_probs,
            'label': 'Malignant' if prediction == 1 else 'Benign',
            'image_tensor_tsfmed': image_tensor_tsfmed
        }

    def _extract_access_no_sop_uid(self, image_path):
        """Extract access_no and sop_uid from image path"""
        # Get filename (without extension) as sop_uid
        sop_uid = os.path.splitext(os.path.basename(image_path))[0]

        # Get parent folder name as access_no
        access_no = os.path.basename(os.path.dirname(image_path))

        return access_no, sop_uid

    def _visualize_prediction(self, transformed_image, output_folder, result, ground_truth=None):
        """Draw prediction results on transformed image and save"""
        # Convert tensor back to PIL image for visualization
        # Note: The tensor is already normalized, so we need to unnormalize it
        inv_normalize = transforms.Normalize(
            mean=[-m/s for m, s in zip(self.m_config['image_normalized_mean'], self.m_config['image_normalized_std'])],
            std=[1/s for s in self.m_config['image_normalized_std']]
        )

        # Remove batch dimension and unnormalize
        image_tensor = inv_normalize(transformed_image.squeeze(0))
        # Convert to PIL image
        image_pil = transforms.ToPILImage()(image_tensor)

        # Get the dimensions of the transformed image
        img_width, img_height = image_pil.size

        # Create a figure with the exact same dimensions as the transformed image
        # Using a DPI of 100 to ensure proper scaling
        dpi = 100
        scale_factor = 2 # Scale factor to improve text/image clarity
        fig_width = img_width / dpi * scale_factor
        fig_height = img_height / dpi * scale_factor

        plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
        plt.imshow(image_pil)
        plt.axis('off')

        # Calculate appropriate font size based on image dimensions
        fontSize = max(8, min(12, img_height // 50))

        # Add prediction result text
        prediction_text = f"Pred={result['label'][:3]}" # Mal:Malignant or Ben:Benign
        confidence_text = f"Probability:{result['confidence']:.4f}"
        pred_correct = True if result['prediction'] == ground_truth else False
        if pred_correct:
            pred_text_color = 'green'
            pred_status = 'Correct'
            output_imageFolder = os.path.join(output_folder, 'pred_correct')
        else:
            pred_text_color = 'red'
            pred_status = 'Error'
            output_imageFolder = os.path.join(output_folder, 'pred_wrong')

        os.makedirs(output_imageFolder, exist_ok=True)
        pred_text = f"{pred_status}:{prediction_text},{result['confidence']:.3f},gt={ground_truth};"

        # Add text to image with adjusted position based on image size
        plt.text(5, fontSize + 5, pred_text, fontsize=fontSize, color=pred_text_color,
                 bbox=dict(facecolor='white', alpha=0.7))

        if 0:
            # If ground truth exists, add it to the image
            if ground_truth is not None:
                ground_truth_text = f"Ground Truth: {'Malignant' if ground_truth == 1 else 'Benign'}"
                plt.text(5, 2 * (fontSize + 5), ground_truth_text, fontsize=fontSize, color='blue',
                        bbox=dict(facecolor='white', alpha=0.7))

        image_path = result.get('image_path', None)
        # If image path is provided, add it to the image for reference
        if 0 and image_path is not None:
            plt.text(5, img_height - 5, os.path.basename(image_path),
                     fontsize=fontSize, color='black', bbox=dict(facecolor='white', alpha=0.7),
                     verticalalignment='bottom')

        # Save image with the exact dimensions of the transformed image
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        imageName=os.path.basename(image_path)
        output_image = os.path.join(output_imageFolder, f'pred_{os.path.splitext(imageName)[0]}.png')
        plt.savefig(output_image, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close()

    def _save_results_and_visualizations(self, predResult, ground_truth,
                                        result_data, correct, output_folder, visualize=True):
        process_start_time = self.m_time_stats.start_processing_timer()

        image_path = predResult['image_path']
        image_name = os.path.basename(image_path)

        access_no, sop_uid = self._extract_access_no_sop_uid(image_path)
        # this will be used in outof this method to save the results --eton@251109
        result_data.append({
            'image_name': image_name,
            'accession_number': access_no,
            'sop_uid': sop_uid,
            'prediction': predResult['prediction'],
            'confidence': predResult['confidence'],
            'ground_truth': ground_truth,
            'correct': correct
        })

        if visualize:
            if not self.process_correct_predictions and correct:
                # Skip visualization if not processing correct predictions --eton@251109
                return
            try:
                self._visualize_prediction(predResult['image_tensor_tsfmed'], output_folder, predResult, ground_truth)
            except Exception as e:
                print(f"Error visualizing {image_name}: {e}")

        self.m_time_stats.end_processing_timer(process_start_time)

    def _save_case_summary(self, result_data, output_folder_root, case_name):
        """Shared method to save case summary CSV and statistics"""
        # Create DataFrame and save to CSV
        df = pd.DataFrame(result_data)
        output_folder = os.path.join(output_folder_root, "cases_summary")

        os.makedirs(output_folder, exist_ok=True)
        csv_path = os.path.join(output_folder, f'{case_name}_predictions.csv')
        df.to_csv(csv_path, index=False)

        # Calculate statistics
        total_images = len(df)
        correct_predictions = df['correct'].sum()
        error_predictions = total_images - correct_predictions
        return total_images, correct_predictions, error_predictions

    def _get_case_folders(self, root_folder):
        """Get all case folders in the root directory"""
        case_folders = []
        for item in os.listdir(root_folder):
            item_path = os.path.join(root_folder, item)
            if os.path.isdir(item_path):
                case_folders.append((item, item_path))
        case_folders.sort()
        return case_folders

    def _process_individual_case(self, case_name, case_path, output_root_folder, visualize):
        """Process a single case folder"""
        #output_folder = os.path.join(output_root_folder, case_name)
        #use root output folder directly, add case_name prefix to files instead
        output_folder = output_root_folder
        os.makedirs(output_folder, exist_ok=True)

        # Start case timer
        case_start_time = time.perf_counter()

        # Process the case folder
        total_images, correct_predictions, error_predictions, result_data = self._process_one_case_images(
            case_path, output_folder, visualize
        )

        # End case timer and record
        case_end_time = time.perf_counter()
        case_time = case_end_time - case_start_time
        self.m_time_stats.record_case_time(case_name, case_time, total_images)

        return case_name, total_images, correct_predictions, error_predictions, result_data

    def _generate_root_summary(self, cases_stats, output_root_folder):
        """Generate summary reports for the root folder"""
        # Separate case statistics and image result data
        cases_data = []
        all_images_data = []

        for case_name, total, correct, error, result_data in cases_stats:
            cases_data.append((case_name, total, correct, error))
            # Add case name to each image result
            for img_data in result_data:
                img_data['case_name'] = case_name
                all_images_data.append(img_data)

        # Calculate overall statistics
        all_total = sum(stat[1] for stat in cases_data)
        all_correct = sum(stat[2] for stat in cases_data)
        all_error = sum(stat[3] for stat in cases_data)

        # Create cases summary CSV
        cases_df = pd.DataFrame(cases_data, columns=['case_name', 'total_images', 'correct_predictions', 'error_predictions'])
        cases_csv_path = os.path.join(output_root_folder, 'cases_summary.csv')
        cases_df.to_csv(cases_csv_path, index=False)

        # Create all images results summary CSV
        if all_images_data:
            all_images_df = pd.DataFrame(all_images_data)
            all_images_csv_path = os.path.join(output_root_folder, 'all_images_predictions.csv')
            all_images_df.to_csv(all_images_csv_path, index=False)

        # Create text summary report
        summary_path = os.path.join(output_root_folder, 'summary_report.txt')
        with open(summary_path, 'w') as f:
            f.write("Prediction Summary Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total cases processed: {len(cases_data)}\n")
            f.write(f"Total images: {all_total}\n")
            f.write(f"Correct predictions: {all_correct}\n")
            f.write(f"Error predictions: {all_error}\n")
            f.write(f"Overall accuracy: {all_correct/all_total*100:.2f}%\n\n")

            f.write("Case Details:\n")
            f.write("-" * 50 + "\n")
            for case_name, total, correct, error in cases_data:
                accuracy = correct/total*100 if total > 0 else 0
                f.write(f"Case: {case_name}\n")
                f.write(f"  Total images: {total}\n")
                f.write(f"  Correct: {correct}\n")
                f.write(f"  Error: {error}\n")
                f.write(f"  Accuracy: {accuracy:.2f}%\n\n")

    def process_single_image(self, image_path, output_folder):
        """Process a single image file and save results"""
        # Start total timer
        self.m_time_stats.start_total_timer()

        if not os.path.exists(image_path):
            print(f"Error: Image file does not exist: {image_path}")
            return

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        # Predict image
        result = self._predict_single_image(image_path)
        if result is None:
            return

        # Extract access_no and sop_uid and get ground truth
        access_no, sop_uid = self._extract_access_no_sop_uid(image_path)
        ground_truth = self._get_ground_truth(access_no, sop_uid)

        if __DEBUG__:
            # Print results to console
            print(f"Prediction: {result['label']}")
            print(f"Malignant Probability: {result['confidence']:.4f}")
            if ground_truth is not None:
                print(f"Ground Truth: {'Malignant' if ground_truth == 1 else 'Benign'}")

        # Save results to text file
        image_name = os.path.basename(image_path)
        output_file = os.path.join(output_folder, f'{os.path.splitext(image_name)[0]}_prediction.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Image: {image_name}\n")
            f.write(f"Prediction: {result['label']}\n")
            f.write(f"Probability: {result['confidence']:.4f}\n")
            if ground_truth is not None:
                f.write(f"Ground Truth: {'Malignant' if ground_truth == 1 else 'Benign'}\n")
                f.write(f"access_no: {access_no}\n")
                f.write(f"sop_uid: {sop_uid}\n")

        # Visualize prediction results and save image
        output_image = os.path.join(output_folder, f'{os.path.splitext(image_name)[0]}_prediction.png')
        # Pass the transformed image tensor instead of image path
        self._visualize_prediction(result['image_tensor_tsfmed'], output_folder, result, ground_truth)

        print(f"Prediction results saved to: {output_file}")
        print(f"Visualization results saved to: {output_image}")

        # End total timer and print statistics
        self.m_time_stats.end_total_timer()
        self.m_time_stats.print_statistics()

    def _process_one_case_images(self, case_folder, output_folder, visualize=True, batch_size=16):
        """Process all images in a single case folder"""

        # Apply case data filter before processing
        filter_passed, filter_results = self.m_caseDataChecker.filter_case(case_folder)

        if not filter_passed:
            case_name = os.path.basename(case_folder)
            print(f"⚠️  Case '{case_name}' failed filter checks:")
            for node_name, node_result in filter_results.get("filter_results", {}).items():
                if not node_result.get("passed", False):
                    message = node_result.get("message", "Filter failed")
                    print(f"   - {node_name}: {message}")

            # Return empty results for filtered cases
            return 0, 0, 0, []
        if __DEBUG__:
            case_name = os.path.basename(case_folder)
            print(f"✅ Case '{case_name}' passed filter checks, processing...")

        os.makedirs(output_folder, exist_ok=True)

        result_data = []

        # Get all image files
        image_extensions = ['.png', '.jpg', '.jpeg']
        image_paths = []
        for file in os.listdir(case_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(case_folder, file))

        # Use batch prediction
        if image_paths:
            pred_results = self._predict_batch_images(image_paths, batch_size)

            # Process results
            for predResult in pred_results:
                try:
                    access_no, sop_uid = self._extract_access_no_sop_uid(predResult['image_path'])
                    ground_truth = self._get_ground_truth(access_no, sop_uid)
                    correct = (predResult['prediction'] == ground_truth) if ground_truth is not None else None

                    # Save results and visualizations
                    self._save_results_and_visualizations(
                        predResult, ground_truth, result_data, correct, output_folder, visualize
                    )

                except Exception as e:
                    print(f"Error processing {predResult['image_path']}: {e}")

        # Save case summary
        total_images, correct_predictions, error_predictions = self._save_case_summary(
            result_data, output_folder, case_name
        )

        return total_images, correct_predictions, error_predictions, result_data


    def process_one_case_folder(self, case_folder, output_folder, visualize=True, batch_size=16):
        """Process all images in a single case folder"""
        return self._process_one_case_images(case_folder, output_folder, visualize, batch_size)

    def process_root_folder(self, root_folder, output_root_folder, visualize=True):
        """Process all case folders in the root directory"""
        # Start total timer
        self.m_time_stats.start_total_timer()

        print(f"Processing root folder: {root_folder} --->>>{output_root_folder}")
        os.makedirs(output_root_folder, exist_ok=True)

        # Get all case folders
        case_folders = self._get_case_folders(root_folder)
        if not case_folders:
            print(f"No case folders found in {root_folder}")
            return

        # Process each case folder
        cases_stats = []
        for case_name, case_path in tqdm(case_folders, desc="Processing cases"):
            #print(f"Processing case: {case_name}")
            try:
                stats = self._process_individual_case(case_name, case_path, output_root_folder, visualize)
                cases_stats.append(stats)
            except Exception as e:
                print(f"Error processing case {case_name}: {e}")
                # Add failed case to stats with zero values
                cases_stats.append((case_name, 0, 0, 0, []))

        # Generate summary reports
        self._generate_root_summary(cases_stats, output_root_folder)

        # End total timer and print statistics
        self.m_time_stats.end_total_timer()
        self.m_time_stats.print_statistics()

        print(f"Root folder processing completed. Results saved to {output_root_folder}")

    @staticmethod
    def summarize_predictions(results_data):
        """Static method to summarize prediction statistics"""
        if isinstance(results_data, pd.DataFrame):
            df = results_data
        else:
            df = pd.DataFrame(results_data)

        total_images = len(df)
        correct_predictions = df['correct'].sum()
        error_predictions = total_images - correct_predictions

        return total_images, correct_predictions, error_predictions

# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Predict nodule images using pretrained EfficientNet model')
    parser.add_argument('input_path', help='Input path (can be a single image file or a folder containing images)')
    parser.add_argument('--output_folder', required=True, help='Output folder for results')
    parser.add_argument('--model_path', help='Model file path (default: path in config)')
    parser.add_argument('--feature_mapping_file', help='Feature mapping file path (default: path in config)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold (default: value in config)')
    parser.add_argument('--spreadsheet_file', required=True, help='Spreadsheet file path containing ground truth labels (.csv, .xls, .xlsx, .ods)')
    parser.add_argument('--process_correct_predictions', type=bool, default=False,
                        help='Whether to process correctly predicted images (default: False)')
    return parser.parse_args()

# Create a new function to handle all input validations
def validate_inputs(args):
    """
    Validate all input parameters and required files/folders
    Returns:
        tuple: (is_valid, message)
    """
    if not os.path.exists(args.input_path):
        return False, f"Input path does not exist: {args.input_path}"
    if not os.path.exists(args.spreadsheet_file):
        return False, f"Spreadsheet file does not exist: {args.spreadsheet_file}"

    try:
        os.makedirs(args.output_folder, exist_ok=True)
    except Exception as e:
        return False, f"Cannot create output folder: {args.output_folder}, {str(e)}"

    model_path = args.model_path or CONFIG['model_path']

    if not os.path.exists(model_path):
        return False, f"Model file does not exist: {model_path}"

    return True, "All inputs validated successfully"

def get_usage():
    s = """Usage:
        python predict_and_human_confirm.py <input_path>
        --spreadsheet_file <spreadsheet_file>
        --output_folder <output_folder>
        [--model_path model.pt]
        [--feature_mapping_file mapp.json]
        [--threshold 0.5]
        [--process_correct_predictions True|False]
        """
    return s
def main():
    args = parse_arguments()

    is_valid, message = validate_inputs(args)
    if not is_valid:
        print(f"Error: {message}")
        print(get_usage())
        return

    predictor = Predictor(
        model_path=args.model_path,
        feature_mapping_file=args.feature_mapping_file,
        threshold=args.threshold,
        spreadsheet_file=args.spreadsheet_file,
        process_correct_predictions=args.process_correct_predictions
    )

    if predictor.model is None:
        print("Program exited due to errors in model loading")
        return

    if os.path.isfile(args.input_path):
        predictor.process_single_image(args.input_path, args.output_folder)
    elif os.path.isdir(args.input_path):
        has_subdirectories = any(os.path.isdir(os.path.join(args.input_path, item))
                              for item in os.listdir(args.input_path))

        if has_subdirectories:
            predictor.process_root_folder(args.input_path, args.output_folder)
        else:
            predictor.time_stats.start_total_timer()
            predictor.process_one_case_folder(args.input_path, args.output_folder)
            predictor.time_stats.end_total_timer()
            predictor.time_stats.print_statistics()
    else:
        print(f"Error: Input path does not exist or is not a valid file/folder: {args.input_path}")

if __name__ == "__main__":
    main()
