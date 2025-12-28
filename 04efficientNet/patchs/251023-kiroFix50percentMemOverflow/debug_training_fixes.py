#!/usr/bin/env python3
"""
Debugging fixes for train_nodule_feature_cnn_model_v75.py memory and performance issues
"""

# 1. MEMORY MANAGEMENT FIXES
def add_memory_management():
    """Add these imports and functions to your training script"""
    
    import gc
    import torch
    
    def clear_gpu_memory():
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def get_gpu_memory_info():
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3     # GB
            return f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"
        return "GPU not available"

# 2. IMPROVED DATALOADER CONFIGURATION
IMPROVED_CONFIG = {
    # Use multiple workers for data loading (but not too many for A10G instance)
    'num_workers': 2,  # Start with 2, can increase to 4 if stable
    
    # Reduce batch size to prevent memory issues
    'batch_size': 16,  # Reduced from 32
    
    # Disable pin_memory if having memory issues
    'pin_memory': False,  # Set to False if memory issues persist
    
    # Add prefetch factor for better performance
    'prefetch_factor': 2,
    
    # Enable persistent workers to avoid recreation overhead
    'persistent_workers': True,
}

# 3. IMPROVED DATASET CLASS WITH CACHING
class ImprovedNoduleDataset:
    """Improved dataset with better error handling and optional caching"""
    
    def __init__(self, df, image_root, feature_mapping, transform=None, cache_images=False):
        self.df = df.copy()
        self.image_root = image_root
        self.feature_mapping = feature_mapping
        self.transform = transform
        self.cache_images = cache_images
        self.image_cache = {} if cache_images else None
        
        # Pre-validate image paths to avoid runtime errors
        self._validate_images()
    
    def _validate_images(self):
        """Pre-validate that images exist"""
        print("Validating image paths...")
        valid_indices = []
        missing_count = 0
        
        for idx, row in self.df.iterrows():
            image_path = os.path.join(self.image_root, row['access_no'], f"{row['sop_uid']}.jpg")
            if os.path.exists(image_path):
                valid_indices.append(idx)
            else:
                missing_count += 1
        
        print(f"Found {len(valid_indices)} valid images, {missing_count} missing")
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_root, row['access_no'], f"{row['sop_uid']}.jpg")
        
        # Use cache if enabled
        if self.cache_images and image_path in self.image_cache:
            image = self.image_cache[image_path]
        else:
            try:
                image = Image.open(image_path).convert('RGB')
                if self.cache_images and len(self.image_cache) < 1000:  # Limit cache size
                    self.image_cache[image_path] = image
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                image = Image.new('RGB', (224, 224), (128, 128, 128))  # Gray instead of black
        
        if self.transform:
            image = self.transform(image)
        
        # ... rest of the method remains the same

# 4. MEMORY-EFFICIENT TRAINING LOOP
def memory_efficient_train_epoch(model, loader, loss_fn, optimizer, device, config=None):
    """Memory-efficient training epoch with explicit cleanup"""
    model.train()
    total_loss = 0
    metrics_calc = DetailedMetricsCalculator_V60(model.mappings)
    metrics_calc.reset()
    
    # Add memory monitoring
    print(f"Starting epoch - {get_gpu_memory_info()}")
    
    for batch_idx, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        # Move to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(batch['image'])
        losses = loss_fn(outputs, batch)
        losses['total'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                     max_norm=config['max_grad_norm'] if config else 1.0)
        optimizer.step()
        
        total_loss += losses['total'].item()
        metrics_calc.update(outputs, batch)
        
        # Explicit cleanup every N batches
        if batch_idx % 50 == 0:
            del outputs, losses
            clear_gpu_memory()
        
        # Memory monitoring for debugging
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx} - {get_gpu_memory_info()}")
    
    return total_loss / len(loader), metrics_calc.compute_metrics()

# 5. DISABLE VISUALIZATION DURING DEBUGGING
def create_minimal_config_for_debugging():
    """Create a minimal config for debugging"""
    return {
        'batch_size': 8,          # Very small batch size
        'num_workers': 1,         # Single worker
        'pin_memory': False,      # Disable pin memory
        'num_epochs': 5,          # Short run for testing
        'verify_epoch_interval': 10,  # Disable periodic verification
        'early_stop_patience': 3, # Quick early stopping
        'disable_visualization': True,  # Skip visualization
        'disable_tensorboard': True,    # Skip tensorboard
    }

# 6. DIAGNOSTIC FUNCTIONS
def diagnose_system_resources():
    """Diagnose system resources before training"""
    import psutil
    import torch
    
    print("=== System Resource Diagnosis ===")
    
    # CPU and Memory
    cpu_count = psutil.cpu_count()
    memory = psutil.virtual_memory()
    print(f"CPU Cores: {cpu_count}")
    print(f"Total RAM: {memory.total / 1024**3:.2f} GB")
    print(f"Available RAM: {memory.available / 1024**3:.2f} GB")
    print(f"RAM Usage: {memory.percent}%")
    
    # GPU
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_props.name}")
        print(f"GPU Memory: {gpu_props.total_memory / 1024**3:.2f} GB")
        print(f"Current GPU Memory: {get_gpu_memory_info()}")
    
    # Disk I/O
    disk = psutil.disk_usage('/')
    print(f"Disk Usage: {disk.percent}%")
    print(f"Free Disk: {disk.free / 1024**3:.2f} GB")

if __name__ == "__main__":
    print("Use these fixes in your training script:")
    print("1. Add memory management functions")
    print("2. Update CONFIG with improved settings")
    print("3. Use ImprovedNoduleDataset")
    print("4. Replace train_epoch with memory_efficient_train_epoch")
    print("5. Run diagnose_system_resources() before training")