import os
import logging
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def setup_logger(model_name):
    """
    Setup logger with both file and console handlers.
    
    Args:
        model_name: Name of the model for organizing log files
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(f"Trainer_{model_name}")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create model directory for logs (same as where model is saved)
    log_dir = os.path.join(os.getcwd(), "saved_models", model_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # File handler (overwrite existing log file)
    log_file = os.path.join(log_dir, f"{model_name}.log")
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logger initialized. Logging to {log_file}")
    
    return logger


def setup_tensorboard(model_name, logger):
    """
    Setup TensorBoard SummaryWriter.
    
    Args:
        model_name: Name of the model for organizing tensorboard logs
        logger: Logger instance for logging messages
        
    Returns:
        SummaryWriter instance
    """
    # Create tensorboard directory for logs
    log_dir = os.path.join(os.getcwd(), "runs", model_name)
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard logging to {log_dir}")
    logger.info(f"To view logs, run: tensorboard --logdir={os.path.join(os.getcwd(), 'runs')}")
    
    return writer


def get_phase_optimizer_params(phase):
    """
    Get learning rate and weight decay for a specific training phase.
    
    Args:
        phase: Training phase (1, 2, 3, or 4)
        
    Returns:
        Tuple of (learning_rate, weight_decay)
    """
    phase_params = {
        1: (2.5e-4, 1e-4),  # Phase 1: lr=2.5e-4, wd=1e-4
        2: (5e-4, 1e-6),     # Phase 2: lr=5e-4, wd=1e-6
        3: (1e-3, 1e-6),     # Phase 3: lr=1e-3, wd=1e-6
        4: (1e-4, 1e-6),     # Phase 4: lr=1e-4, wd=1e-6
    }
    return phase_params.get(phase, (1e-4, 1e-6))


def should_switch_phase(epoch, current_phase):
    """
    Determine if phase should be switched based on current epoch.
    
    Args:
        epoch: Current epoch number (0-indexed)
        current_phase: Current training phase
        
    Returns:
        New phase number if should switch, None otherwise
    """
    # Epoch ranges (0-indexed):
    # Phase 1: epochs 0-4 (epochs 1-5 in 1-indexed)
    # Phase 2: epochs 5-9 (epochs 6-10 in 1-indexed)
    # Phase 3: epochs 10-299 (epochs 11-300 in 1-indexed)
    # Phase 4: epochs 300+ (epochs 301+ in 1-indexed)
    
    if 5 <= epoch < 10 and current_phase != 2:
        return 2
    elif 10 <= epoch < 300 and current_phase != 3:
        return 3
    elif epoch >= 300 and current_phase != 4:
        return 4
    return None


def get_gamma_for_epoch(epoch):
    """
    Calculate gamma values for target spatial spectrum based on current epoch.
    
    Args:
        epoch: Current epoch number (0-indexed)
        
    Returns:
        List of 3 gamma values [gamma1, gamma2, gamma3]
    
    Schedule:
        - Epochs 0-19 (1-20 in 1-indexed): [16, 6, 2.5]
        - Epochs 20-44 (21-45 in 1-indexed): Linear decay to [2.5, 2.5, 2.5]
        - Epochs 45-299 (46-300 in 1-indexed): [2.5, 2.5, 2.5]
        - Epochs 300-319 (301-320 in 1-indexed): Linear decay to [1.0, 1.0, 1.0]
        - Epochs 320+ (321+ in 1-indexed): [1.0, 1.0, 1.0]
    """
    # Initial and final gamma values
    gamma_initial = [16.0, 6.0, 2.5]
    gamma_final = [2.5, 2.5, 2.5]
    
    # Epochs before decay starts (0-19)
    if epoch < 20:
        return gamma_initial
    
    # Linear decay phase (20-44)
    elif 20 <= epoch < 45:
        # Calculate progress through decay period (0 to 1)
        decay_start = 20
        decay_end = 45
        progress = (epoch - decay_start) / (decay_end - decay_start)
        
        # Linear interpolation for each gamma value
        gamma = [
            gamma_initial[i] - (gamma_initial[i] - gamma_final[i]) * progress
            for i in range(3)
        ]
        return gamma

    # Epochs after decay ends (45-299)
    elif 45 <= epoch < 300:
        return gamma_final
    
    # Second decay phase (300-319)
    elif 300 <= epoch < 320:
        decay_start = 300
        decay_end = 320
        progress = (epoch - decay_start) / (decay_end - decay_start)

        gamma_start = [2.5, 2.5, 2.5]
        gamma_target = [1.0, 1.0, 1.0]
        
        # Linear interpolation for each gamma value
        gamma = [
            gamma_start[i] - (gamma_start[i] - gamma_target[i]) * progress
            for i in range(3)
        ]
        return gamma
    
    # Final phase (320+)
    else:
        return [1.0, 1.0, 1.0]


def compute_bce_loss(epoch, pred, target, vad, alpha=10.0):
    """
    Calculate Binary Cross-Entropy loss with optional weighted loss based on gamma values.
    
    Args:
        epoch: Current epoch number (for gamma scheduling)
        pred: Predicted output tensor (batch_size, num_outputs, ...)
        target: Target tensor (batch_size, num_outputs, ...)
        vad: Voice Activity Detection mask (batch_size, num_microphones, time_frames)
        alpha: Weight multiplier for target values in weighted loss (default: 10.0)
        
    Returns:
        Total loss value
    """
    vad_mask = vad[:, 0, :]
    vad_mask = vad_mask.unsqueeze(1).unsqueeze(1)
    
    criterion = nn.BCELoss()
    total_loss = 0
    num_outputs = pred.shape[1]

    # Check if we should use weighted loss (when gamma is [1.0, 1.0, 1.0])
    if get_gamma_for_epoch(epoch) == [1.0, 1.0, 1.0]:
        # Weighted BCE loss
        for i in range(num_outputs):
            layer_pred = pred[:, i, ...].unsqueeze(1)
            layer_target = target[:, i, ...].unsqueeze(1)
    
            loss_weights = 1.0 + (alpha * layer_target)

            vad_expanded = vad_mask.expand_as(layer_pred)
            
            active_pred = layer_pred[vad_expanded == 1]
            active_target = layer_target[vad_expanded == 1]
            active_weights = loss_weights[vad_expanded == 1]
            
            layer_loss = F.binary_cross_entropy(
                active_pred, 
                active_target, 
                weight=active_weights, 
                reduction='mean'
            )
        
            total_loss += layer_loss
    else:
        # Standard BCE loss
        for i in range(num_outputs):
            layer_pred = pred[:, i, ...].unsqueeze(1)
            layer_target = target[:, i, ...].unsqueeze(1)
        
            vad_expanded = vad_mask.expand_as(layer_pred)
            
            masked_pred = layer_pred[vad_expanded == 1]
            masked_target = layer_target[vad_expanded == 1]
            
            loss = criterion(masked_pred, masked_target)
            total_loss += loss
    
    return total_loss


def log_training_config(logger, config):
    """
    Log a summary of all training parameters and configuration.
    
    Args:
        logger: Logger instance
        config: Dictionary containing training configuration parameters
    """
    logger.info("="*60)
    logger.info("TRAINING CONFIGURATION SUMMARY")
    logger.info("="*60)
    
    # Model configuration
    logger.info("Model Configuration:")
    logger.info(f"  Model Name: {config['model_name']}")
    logger.info(f"  Model Version: {config['model_version']}")
    logger.info(f"  MPE Type: {config['mpe_type']}")
    logger.info(f"  Device: {config['device']}")
    
    # Dataset configuration
    logger.info("\nDataset Configuration:")
    logger.info(f"  Data Path: {config['data_path']}")
    logger.info(f"  Mode: {config['mode']}")
    logger.info(f"  Max Microphones: {config['max_microphones']}")
    logger.info(f"  Sample Rate: {config['sample_rate']} Hz")
    logger.info(f"  Noise Probability: {config['noise_probability']}")
    logger.info(f"  Initial Training Phase: {config['training_phase']}")
    logger.info(f"  Total Samples: {config['total_samples']}")
    
    # Training hyperparameters
    logger.info("\nTraining Hyperparameters:")
    logger.info(f"  Total Epochs: {config['epochs']}")
    logger.info(f"  Batch Size: {config['batch_size']}")
    logger.info(f"  Initial Learning Rate: {config['initial_lr']}")
    logger.info(f"  Initial Weight Decay: {config['initial_wd']}")
    logger.info(f"  Optimizer: AdamW")
    logger.info(f"  LR Scheduler: ReduceLROnPlateau (factor=0.9, patience=2)")
    
    logger.info("="*60)
    logger.info("")
