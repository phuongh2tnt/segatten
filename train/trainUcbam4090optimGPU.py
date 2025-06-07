import torch
from tqdm import tqdm
import numpy as np
import argparse
from torch.optim import Adam, AdamW
import utils.metrics as metrics
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from utils.metric2 import accuracy, iou, f1, precision, recall 
from torch.amp import GradScaler, autocast
import gc
import time
# Import your UNet class
from cbamunet import UNet  

# Setup CUDA với tối ưu hóa
def setup_cuda():
    seed = 50
    # Tối ưu hóa CUDA
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # Tự động tìm thuật toán tối ưu
    torch.backends.cudnn.deterministic = False  # Tăng tốc độ, giảm reproducibility
    
    # Memory management
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95)  # Sử dụng 95% GPU memory
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Seed cho tất cả GPU
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Hiển thị thông tin GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    
    return device

def check_gpu_utilization():
    """Kiểm tra GPU utilization"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")

def train_model(accumulation_steps=4):  # Tăng accumulation steps
    model.train()
    train_loss = 0.0
    train_metrics = {'iou': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    scaler = GradScaler('cuda')

    optimizer.zero_grad()
    
    # Tối ưu hóa loop với non_blocking transfer
    for i, (img, gt) in enumerate(tqdm(train_loader, ncols=80, desc='Training')):
        # Non-blocking transfer để tăng tốc
        img = img.to(device, dtype=torch.float, non_blocking=True)
        gt = gt.to(device, dtype=torch.long, non_blocking=True)
        
        with autocast('cuda'):
            logits = model(img)
            loss = loss_fn(logits, gt) / accumulation_steps
        
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            # Gradient clipping để tránh gradient exploding
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        train_loss += loss.item() * accumulation_steps
        
        # Tối ưu metric calculation - chỉ tính trên một phần dữ liệu
        if i % 10 == 0:  # Chỉ tính metrics mỗi 10 batch để tăng tốc
            with torch.no_grad():
                prediction = logits.argmax(axis=1).cpu().numpy()
                gt_cpu = gt.cpu().numpy()
                train_metrics['iou'] += iou(prediction, gt_cpu)
                train_metrics['accuracy'] += accuracy(prediction, gt_cpu)
                train_metrics['precision'] += precision(prediction, gt_cpu)
                train_metrics['recall'] += recall(prediction, gt_cpu)
                train_metrics['f1'] += f1(prediction, gt_cpu)
        
        # Memory cleanup mỗi 50 iterations
        if i % 50 == 0:
            torch.cuda.empty_cache()

    # Normalize metrics
    metric_batches = len(train_loader) // 10 + 1
    for key in train_metrics:
        train_metrics[key] /= metric_batches

    return train_loss / len(train_loader), train_metrics

def validate_model():
    model.eval()
    valid_loss = 0.0
    val_metrics = {'iou': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    
    with torch.no_grad():
        for i, (img, gt) in enumerate(tqdm(valid_loader, ncols=80, desc='Validating')):
            # Non-blocking transfer
            img = img.to(device, dtype=torch.float, non_blocking=True)
            gt = gt.to(device, dtype=torch.long, non_blocking=True)
            
            with autocast('cuda'):
                logits = model(img)
                loss = loss_fn(logits, gt)
            
            valid_loss += loss.item()
            
            # Tính metrics
            prediction = logits.argmax(axis=1).cpu().numpy()
            gt_cpu = gt.cpu().numpy()
            val_metrics['iou'] += iou(prediction, gt_cpu)
            val_metrics['accuracy'] += accuracy(prediction, gt_cpu)
            val_metrics['precision'] += precision(prediction, gt_cpu)
            val_metrics['recall'] += recall(prediction, gt_cpu)
            val_metrics['f1'] += f1(prediction, gt_cpu)
            
            # Memory cleanup
            if i % 20 == 0:
                torch.cuda.empty_cache()

    for key in val_metrics:
        val_metrics[key] /= len(valid_loader)

    return valid_loss / len(valid_loader), val_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a deep model for shrimp segmentation')
    parser.add_argument('-d', '--dataset', default="E:/thanh/ntu_group/phuong/segatten/train/dataset", type=str, help='Dataset folder')
    parser.add_argument('-e', '--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('-b', '--batch-size', default=8, type=int, help='Batch size (tăng từ 4 lên 8)')
    parser.add_argument('-i', '--img-size', default=480, type=int, help='Image size')
    parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, help='Checkpoint folder')
    parser.add_argument('-t', '--metric', default='iou', type=str, help='Metric for optimization')

    cmd_args = parser.parse_args()
    device = setup_cuda()

    from utils.lanedatasetv2 import LaneDataset

    # Tối ưu DataLoader
    train_dataset = LaneDataset(dataset_dir=cmd_args.dataset, subset='test', img_size=cmd_args.img_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cmd_args.batch_size,
        shuffle=True,
        num_workers=8,  # Tăng từ 6 lên 8
        pin_memory=True,  # Tăng tốc transfer GPU
        persistent_workers=True,  # Giữ workers alive
        prefetch_factor=4  # Prefetch nhiều batches
    )

    valid_dataset = LaneDataset(dataset_dir=cmd_args.dataset, subset='valid', img_size=cmd_args.img_size)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cmd_args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    # Model với compile để tăng tốc (PyTorch 2.0+)
    model = UNet(
        in_channels=3,  
        out_channels=2  
    ).to(device)
    
    # Compile model nếu PyTorch >= 2.0
    try:
        model = torch.compile(model, mode='max-autotune')
        print("Model compiled successfully!")
    except:
        print("Model compilation not available, using standard model")

    # Tối ưu loss function
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
    
    # Tối ưu optimizer
    optimizer = AdamW(  # AdamW thường tốt hơn Adam
        model.parameters(), 
        lr=0.001,
        weight_decay=1e-4,  # L2 regularization
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=cmd_args.epochs,
        steps_per_epoch=len(train_loader)
    )
    
    train_history = {'loss': [], 'iou': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    val_history = {'loss': [], 'iou': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    
    # Training loop với timing
    max_perf = 0
    start_time = time.time()
    
    print("=== Starting Training ===")
    check_gpu_utilization()
    
    for epoch in range(cmd_args.epochs):
        epoch_start = time.time()
        
        train_loss, train_metrics = train_model()
        val_loss, val_metrics = validate_model()

        train_perf = train_metrics[cmd_args.metric]
        valid_perf = val_metrics[cmd_args.metric]
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        print(f'Epoch: {epoch} | Time: {epoch_time:.1f}s | LR: {current_lr:.6f}')
        print(f'Train {cmd_args.metric}: {train_perf:.4f} | Valid {cmd_args.metric}: {valid_perf:.4f}')
        
        # Save metrics
        train_history['loss'].append(train_loss)
        val_history['loss'].append(val_loss)
        train_history['iou'].append(train_metrics['iou'])
        val_history['iou'].append(val_metrics['iou'])
        train_history['f1'].append(train_metrics['f1'])
        val_history['f1'].append(val_metrics['f1'])
        train_history['precision'].append(train_metrics['precision'])
        val_history['precision'].append(val_metrics['precision'])
        train_history['recall'].append(train_metrics['recall'])
        val_history['recall'].append(val_metrics['recall'])
        train_history['accuracy'].append(train_metrics['accuracy'])
        val_history['accuracy'].append(val_metrics['accuracy'])

        # Save best model
        path = "E:/thanh/ntu_group/phuong/segatten/train/checkpoints"
        path2 = "E:/thanh/ntu_group/phuong/segatten/train/graph"
        
        if valid_perf > max_perf:
            print(f'Valid {cmd_args.metric} increased ({max_perf:.4f} --> {valid_perf:.4f}). Model saved')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'max_perf': valid_perf
            }, f"{path}/unetcbam_epoch_{epoch}_{cmd_args.metric}_{valid_perf:.4f}.pt")
            max_perf = valid_perf
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        # Check GPU utilization every 10 epochs
        if epoch % 10 == 0:
            check_gpu_utilization()
    
    total_time = time.time() - start_time
    print(f"\n=== Training Completed ===")
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Average time per epoch: {total_time/cmd_args.epochs:.1f} seconds")
    
    # Plot and save training and validation metrics
    epochs_range = range(cmd_args.epochs)
    for metric_name in train_history:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, train_history[metric_name], label=f'Training {metric_name}', linewidth=2)
        plt.plot(epochs_range, val_history[metric_name], label=f'Validation {metric_name}', linewidth=2)
        plt.xlabel('Epochs')
        plt.ylabel(metric_name.capitalize())
        plt.title(f'{metric_name.capitalize()} vs. Epochs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{path2}/unetcbam_{metric_name}.png", dpi=300, bbox_inches='tight')
        plt.close()  # Đóng figure để tiết kiệm memory
    
    print("All plots saved successfully!")
