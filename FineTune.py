import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from LibriSpeechDataset import LibriSpeechDataset
from model.main import GI_DOAEnet
import logging
import sys


torch.manual_seed(42)


class Trainer:
    def __init__(self, model_name, data_path, mode, device, max_microphones, sample_rate, MPE_type, batch_size, lr, epochs, resume_from_checkpoint=False):
        self.model_name = model_name
        self.data_path = data_path
        self.mode = mode
        self.device = device
        self.max_microphones = max_microphones
        self.sample_rate = sample_rate
        self.MPE_type = MPE_type
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.resume_from_checkpoint = resume_from_checkpoint

        self.training_phase = 1
        self.start_epoch = 0
        self.min_loss = torch.inf
        
        # Setup logger
        self._setup_logger()
        
        # Setup TensorBoard writer
        self._setup_tensorboard()

        self.model = self.load_model()

        # Initialize optimizer with Phase 1 parameters
        lr, weight_decay = self._get_phase_optimizer_params(self.training_phase)
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler - reduces LR by 0.9 if loss doesn't improve for 2 epochs
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.9, 
            patience=2
        )
        
        self.logger.info(f"Initialized with Phase {self.training_phase} (LR={lr}, Weight Decay={weight_decay})")

        # Create dataset instance
        self.dataset = LibriSpeechDataset(
            data_path=data_path,
            mode=mode,
            device=device,
            max_microphones=max_microphones,
            sample_rate=sample_rate,
            training_phase=self.training_phase,
            logger=self.logger,
        )

        # Load checkpoint if resuming training
        if self.resume_from_checkpoint:
            self.load_checkpoint()
    
    def _setup_logger(self):
        """Setup logger with both file and console handlers."""
        # Create logger
        self.logger = logging.getLogger(f"Trainer_{self.model_name}")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Create model directory for logs (same as where model is saved)
        log_dir = os.path.join(os.getcwd(), "saved_models", self.model_name)
        os.makedirs(log_dir, exist_ok=True)
        
        # File handler
        log_file = os.path.join(log_dir, f"{self.model_name}_training.log")
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logger initialized. Logging to {log_file}")
    
    def _setup_tensorboard(self):
        """Setup TensorBoard SummaryWriter."""
        # Create tensorboard directory for logs
        log_dir = os.path.join(os.getcwd(), "runs", self.model_name)
        os.makedirs(log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=log_dir)
        self.logger.info(f"TensorBoard logging to {log_dir}")
        self.logger.info(f"To view logs, run: tensorboard --logdir={os.path.join(os.getcwd(), 'runs')}")
    
    def load_model(self):
        self.logger.info("Loading pretrained model...")
        model_path = os.path.join(os.getcwd(), "pretrained", "GI_DOAEnet_{}.tar".format(self.MPE_type))
        pretrained = torch.load(model_path, map_location='cpu')
        model = GI_DOAEnet(MPE_type=self.MPE_type)
        model.load_state_dict(pretrained, strict=True)  
        model.load_layers()
        
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze only the new elevation_mapping_layers
        for param in model.SSMBs_elvation.parameters():
            param.requires_grad = True
        
        self.logger.info("Frozen all layers except SSMBs_elvation for fine-tuning")
        model.to(self.device)
        return model
    
    def load_checkpoint(self):
        """Load checkpoint to resume training from a previous state."""
        checkpoint_path = os.path.join(os.getcwd(), "saved_models", self.model_name, f"{self.model_name}.tar")
        
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
            return
        
        self.logger.info(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info("Loaded model state from checkpoint")
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info("Loaded optimizer state from checkpoint")
        
        # Load scheduler state
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.logger.info("Loaded scheduler state from checkpoint")
        
        # Load training metadata
        self.start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        self.training_phase = checkpoint['training_phase']
        self.min_loss = checkpoint['min_loss']
        
        # Update dataset training phase
        self.dataset.set_training_phase(self.training_phase)
        
        self.logger.info(f"Resuming from epoch {self.start_epoch}")
        self.logger.info(f"Training phase: {self.training_phase}")
        self.logger.info(f"Min loss so far: {self.min_loss:.4f}")
    
    def set_training_phase(self, phase):
        """
        Switch to a different training phase and reinitialize optimizer.
        
        Args:
            phase: New phase (1, 2, or 3)
        """
        self.training_phase = phase
        self.dataset.set_training_phase(phase)
        
        # Get phase-specific optimizer parameters
        lr, weight_decay = self._get_phase_optimizer_params(phase)
        
        # Reinitialize optimizer with new parameters
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Reinitialize scheduler with new optimizer
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.9,
            patience=2
        )
        
        self.logger.info(f"Switched to training phase {phase} (LR={lr}, Weight Decay={weight_decay})")
    
    @staticmethod
    def _get_phase_optimizer_params(phase):
        """
        Get learning rate and weight decay for a specific phase.
        
        Args:
            phase: Training phase (1, 2, or 3)
            
        Returns:
            Tuple of (learning_rate, weight_decay)
        """
        phase_params = {
            1: (2.5e-4, 1e-4),  # Phase 1: lr=2.5e-4, wd=1e-4
            2: (5e-4, 1e-6),     # Phase 2: lr=5e-4, wd=1e-6
            3: (1e-3, 1e-6),     # Phase 3: lr=1e-3, wd=1e-6
        }
        return phase_params.get(phase, (1e-4, 1e-6))
    
    def _should_switch_phase(self, epoch):
        """
        Determine if phase should be switched based on current epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
            
        Returns:
            New phase number if should switch, None otherwise
        """
        # Epoch ranges (0-indexed):
        # Phase 1: epochs 0-4 (epochs 1-5 in 1-indexed)
        # Phase 2: epochs 5-9 (epochs 6-10 in 1-indexed)
        # Phase 3: epochs 10+ (epochs 11+ in 1-indexed)
        
        if epoch == 5 and self.training_phase == 1:
            return 2
        elif epoch == 10 and self.training_phase == 2:
            return 3
        return None
    
    @staticmethod
    def _get_gamma_for_epoch(epoch):
        """
        Calculate gamma values for target spatial spectrum based on current epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
            
        Returns:
            List of 3 gamma values [gamma1, gamma2, gamma3]
        
        Schedule:
            - Epochs 0-19 (1-20 in 1-indexed): [16, 6, 2.5]
            - Epochs 20-44 (21-45 in 1-indexed): Linear decay to [2.5, 2.5, 2.5]
            - Epochs 45+ (46+ in 1-indexed): [2.5, 2.5, 2.5]
        """
        # Initial and final gamma values
        gamma_initial = [16.0, 6.0, 2.5]
        gamma_final = [2.5, 2.5, 2.5]
        
        # Epochs before decay starts (0-19)
        if epoch < 20:
            return gamma_initial
        
        # Epochs after decay ends (45+)
        elif epoch >= 45:
            return gamma_final
        
        # Linear decay phase (20-44)
        else:
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
    
    def BCE_loss(self, pred, target, vad):
        vad_mask = vad[:, 0, :]
        vad_mask = vad_mask.unsqueeze(1).unsqueeze(1)
        
        criterion = nn.BCELoss()
        total_loss = 0
        
        num_outputs = pred.shape[1]
        for i in range(num_outputs):
            layer_pred = pred[:, i, ...].unsqueeze(1)
            layer_target = target[:, i, ...].unsqueeze(1)
            
            vad_expanded = vad_mask.expand_as(layer_pred)
            
            masked_pred = layer_pred[vad_expanded == 1]
            masked_target = layer_target[vad_expanded == 1]
            
            loss = criterion(masked_pred, masked_target)
            total_loss += loss
        
        return total_loss

    def train_one_epoch(self, epoch):
        self.model.train()

        self.dataset.sample_num_microphones()

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        print_every = len(dataloader) // 5
        total_loss = 0
        total_az_loss = 0
        total_el_loss = 0
        
        for batch_idx, (audio, vad, room_params) in enumerate(dataloader):
            self.dataset.sample_num_microphones()
            if batch_idx != 0 and batch_idx % print_every == 0:
                average_loss = total_loss / (batch_idx + 1)
                average_az_loss = total_az_loss / (batch_idx + 1)
                average_el_loss = total_el_loss / (batch_idx + 1)
                self.logger.info(f"| Epoch {epoch} | Average Loss: {average_loss:.4f} | Average AZ Loss: {average_az_loss:.4f} | Average EL Loss: {average_el_loss:.4f} |")

            audio = audio.to(self.device)
            vad = vad.to(self.device)
            mic_coordinates = room_params['mic_coords'].to(self.device)
            source_coordinates = room_params['source_coords'].to(self.device)

            self.optimizer.zero_grad()

            x_out_az, x_out_el, target_az, target_el, vad_framed = self.model(audio, mic_coordinates, vad, source_coordinates, return_target=True)

            loss_az = self.BCE_loss(x_out_az, target_az, vad_framed)
            loss_el = self.BCE_loss(x_out_el, target_el, vad_framed)
            loss = loss_el
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()

            total_loss += loss.item()
            total_az_loss += loss_az.item()
            total_el_loss += loss_el.item()

        avg_total_loss = total_loss / len(dataloader)
        avg_az_loss = total_az_loss / len(dataloader)
        avg_el_loss = total_el_loss / len(dataloader)
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/Total', avg_total_loss, epoch)
        self.writer.add_scalar('Loss/Azimuth', avg_az_loss, epoch)
        self.writer.add_scalar('Loss/Elevation', avg_el_loss, epoch)
        
        return avg_total_loss
    
    def _log_training_config(self):
        """Log a summary of all training parameters and configuration."""
        self.logger.info("="*60)
        self.logger.info("TRAINING CONFIGURATION SUMMARY")
        self.logger.info("="*60)
        
        # Model configuration
        self.logger.info("Model Configuration:")
        self.logger.info(f"  Model Name: {self.model_name}")
        self.logger.info(f"  MPE Type: {self.MPE_type}")
        self.logger.info(f"  Device: {self.device}")
        
        # Dataset configuration
        self.logger.info("\nDataset Configuration:")
        self.logger.info(f"  Data Path: {self.data_path}")
        self.logger.info(f"  Mode: {self.mode}")
        self.logger.info(f"  Max Microphones: {self.max_microphones}")
        self.logger.info(f"  Sample Rate: {self.sample_rate} Hz")
        self.logger.info(f"  Initial Training Phase: {self.training_phase}")
        self.logger.info(f"  Total Samples: {len(self.dataset)}")
        
        # Training hyperparameters
        self.logger.info("\nTraining Hyperparameters:")
        self.logger.info(f"  Total Epochs: {self.epochs}")
        self.logger.info(f"  Batch Size: {self.batch_size}")
        initial_lr, initial_wd = self._get_phase_optimizer_params(self.training_phase)
        self.logger.info(f"  Initial Learning Rate: {initial_lr}")
        self.logger.info(f"  Initial Weight Decay: {initial_wd}")
        self.logger.info(f"  Optimizer: Adam")
        self.logger.info(f"  LR Scheduler: ReduceLROnPlateau (factor=0.9, patience=2)")
        
        self.logger.info("="*60)
        self.logger.info("")

    def __call__(self):
        # Log comprehensive training configuration
        self._log_training_config()
        
        self.logger.info(f"Starting training of {self.model_name} on {self.device}...")

        model_dir = os.path.join(os.getcwd(), "saved_models", self.model_name)
        os.makedirs(model_dir, exist_ok=True)
        checkpoint_path = os.path.join(model_dir, f"{self.model_name}.tar")

        for epoch in range(self.start_epoch, self.epochs):
            # Automatic phase switching based on epoch
            new_phase = self._should_switch_phase(epoch)
            if new_phase is not None:
                self.logger.info(f"{'='*50}")
                self.logger.info(f"Auto-switching to Phase {new_phase} at epoch {epoch}")
                self.logger.info(f"{'='*50}\n")
                self.set_training_phase(new_phase)
            
            # Update gamma values based on epoch
            gamma = self._get_gamma_for_epoch(epoch)
            self.model.set_gamma(gamma)
            
            # Log gamma values to TensorBoard
            self.writer.add_scalar('Hyperparameters/Gamma_1', gamma[0], epoch)
            self.writer.add_scalar('Hyperparameters/Gamma_2', gamma[1], epoch)
            self.writer.add_scalar('Hyperparameters/Gamma_3', gamma[2], epoch)
            
            self.logger.info("-" * 40)
            self.logger.info(f"Epoch {epoch} | Gamma: [{gamma[0]:.2f}, {gamma[1]:.2f}, {gamma[2]:.2f}]")
            
            # Track LR before scheduler step
            lr_before = self.optimizer.param_groups[0]['lr']
            
            loss = self.train_one_epoch(epoch)
            self.logger.info(f"Epoch {epoch}, Loss: {loss:.4f}")

            if loss < self.min_loss:
                self.min_loss = loss
                
                # Save checkpoint when achieving new best loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'training_phase': self.training_phase,
                    'min_loss': self.min_loss,
                }
                torch.save(checkpoint, checkpoint_path)
                self.logger.info(f"Saved checkpoint to: {checkpoint_path}")
            
            # Update learning rate based on loss
            self.scheduler.step(loss)
            
            # Log LR only if it changed
            lr_after = self.optimizer.param_groups[0]['lr']
            if lr_after != lr_before:
                self.logger.info(f"Learning rate changed: {lr_before:.6f} -> {lr_after:.6f}")
            
            # Log learning rate to TensorBoard
            self.writer.add_scalar('Hyperparameters/Learning_Rate', lr_after, epoch)
            
            # Log training phase to TensorBoard
            self.writer.add_scalar('Hyperparameters/Training_Phase', self.training_phase, epoch)


if __name__ == "__main__":
    model_name = "GI_DOAEnet_fine_tuned"
    data_path = os.path.join(os.getcwd(), "data")
    mode = "validation"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_microphones = 12
    sample_rate = 16000
    MPE_type = "PM"
    batch_size = 16
    lr = 1e-4
    epochs = 300
    resume_from_checkpoint = False

    trainer = Trainer(
        model_name=model_name,
        data_path=data_path,
        mode=mode,
        device=device,
        max_microphones=max_microphones,
        sample_rate=sample_rate,
        MPE_type=MPE_type,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    
    trainer()
    
    # Close TensorBoard writer
    trainer.writer.close()
