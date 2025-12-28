#!/usr/bin/env python3
"""
System diagnostics to run before training
"""

import torch
import psutil
import os
import pandas as pd

def check_system_resources():
    """Check system resources and potential bottlenecks"""
    print("=== AWS EC2 A10G System Check ===\n")
    
    # 1. GPU Check
    print("1. GPU Status:")
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"   ✅ GPU: {gpu_props.name}")
        print(f"   ✅ Total VRAM: {gpu_props.total_memory / 1024**3:.2f} GB")
        
        # Test GPU memory allocation
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            print("   ✅ GPU memory allocation test passed")
        except Exception as e:
            print(f"   ❌ GPU memory test failed: {e}")
    else:
        print("   ❌ CUDA not available")
    
    # 2. System Memory
    print("\n2. System Memory:")
    memory = psutil.virtual_memory()
    print(f"   Total RAM: {memory.total / 1024**3:.2f} GB")
    print(f"   Available RAM: {memory.available / 1024**3:.2f} GB")
    print(f"   Memory Usage: {memory.percent}%")
    
    if memory.percent > 80:
        print("   ⚠️  WARNING: High memory usage detected")
    
    # 3. CPU
    print("\n3. CPU:")
    cpu_count = psutil.cpu_count()
    cpu_usage = psutil.cpu_percent(interval=1)
    print(f"   CPU Cores: {cpu_count}")
    print(f"   CPU Usage: {cpu_usage}%")
    
    # 4. Disk I/O
    print("\n4. Disk Status:")
    disk = psutil.disk_usage('/')
    print(f"   Disk Usage: {disk.percent}%")
    print(f"   Free Space: {disk.free / 1024**3:.2f} GB")
    
    if disk.percent > 90:
        print("   ⚠️  WARNING: Low disk space")
    
    # 5. Check data paths
    print("\n5. Data Path Check:")
    data_paths = [
        '/data/dataInTrain/251016-efficientNet/dataset_table/train/all_matched_sops_ds_v3_with_tr13_1016.csv',
        '/data/dataInTrain/251016-efficientNet/dataset_table/val/all_verify_sop_with_predictions.csv',
        '/data/dataInTrain/251016-efficientNet/dataset_images/us_images'
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            if path.endswith('.csv'):
                try:
                    df = pd.read_csv(path)
                    print(f"   ✅ {path} ({len(df)} records)")
                except Exception as e:
                    print(f"   ❌ {path} (read error: {e})")
            else:
                print(f"   ✅ {path} (directory exists)")
        else:
            print(f"   ❌ {path} (not found)")
    
    # 6. Recommendations
    print("\n6. Recommendations:")
    
    if memory.available / 1024**3 < 8:
        print("   ⚠️  Consider reducing batch_size due to low available RAM")
    
    if cpu_count >= 4:
        print("   ✅ Can use num_workers=2-4 for DataLoader")
    else:
        print("   ⚠️  Use num_workers=1 due to limited CPU cores")
    
    print("\n=== Check Complete ===")

if __name__ == "__main__":
    check_system_resources()