import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from LibriSpeechDataset import LibriSpeechDataset
from model.main import GI_DOAEnet
import util


torch.manual_seed(42)


class Trainer:
    def __init__(
        self,
        model_name,
        data_path,
        mode,
        device,
        max_microphones,
        sample_rate,
        MPE_type,
        batch_size,
        lr,
        epochs,
        noise_probability,
        start_epoch=None,
        model_version='v1',
        resume_from_checkpoint=False,
        checkpoint_name=None
    ):
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
        self.noise_probability = noise_probability
        self.start_epoch = start_epoch
        self.model_version = model_version
        self.resume_from_checkpoint = resume_from_checkpoint
        self.checkpoint_name = checkpoint_name

        self.training_phase = 1
        self.min_loss = torch.inf
        
        # Setup logger and TensorBoard writer
        self.logger = util.setup_logger(self.model_name)
        self.writer = util.setup_tensorboard(self.model_name, self.logger)

        self.model = self.load_model()

        # Initialize optimizer with Phase 1 parameters
        lr, weight_decay = util.get_phase_optimizer_params(self.training_phase)
        self.optimizer = optim.AdamW(
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
            noise_probability=self.noise_probability
        )

        # Load checkpoint if resuming training
        if self.resume_from_checkpoint:
            self.load_checkpoint()
    

    def freeze_layers(self, model, layer_names):
        for layer in layer_names:
            for param in getattr(model, layer).parameters():
                param.requires_grad = False

    def load_model(self):
        self.logger.info(f"Loading pretrained model (version: {self.model_version})...")
        model_path = os.path.join(os.getcwd(), "pretrained", "GI_DOAEnet_{}.tar".format(self.MPE_type))
        pretrained = torch.load(model_path, map_location='cpu')
        
        # Initialize model with version specification
        model = GI_DOAEnet(MPE_type=self.MPE_type, model_version=self.model_version)
        
        if self.model_version == 'v1':
            # V1: Load entire pretrained model
            # model.load_state_dict(pretrained, strict=True)
            # self.logger.info("Loaded entire pretrained model (v1)")
            
            # Freeze only CIFE parameters for fine-tuning
            # layers_to_freeze = ['STFT', 'CIFE', 'MPE', 'STDPBs']
            layers_to_freeze = []
        
        else:  # v2
            # V2: Load only CIFE (Channel_invariant_feature_extractor) since architecture differs (RoPE vs MPE)
            # cife_state_dict = {k.replace('CIFE.', ''): v for k, v in pretrained.items() if k.startswith('CIFE.')}
            # model.CIFE.load_state_dict(cife_state_dict, strict=True)
            # self.logger.info("Loaded CIFE (Channel Independent Feature Extractor) from pretrained model")
            
            # Do not freeze any layers for V2 model
            layers_to_freeze = []
        
        self.freeze_layers(model, layers_to_freeze)
        
        self.logger.info(f"Frozen {', '.join(layers_to_freeze) if len(layers_to_freeze) > 0 else 'None'}. All other layers are trainable.")
        
        model.to(self.device)
        return model
    
    def load_checkpoint(self):
        """Load checkpoint to resume training from a previous state."""
        checkpoint_path = os.path.join(os.getcwd(), "saved_models", self.checkpoint_name, f"{self.checkpoint_name}.tar")
        
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
        # self.start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        self.training_phase = checkpoint['training_phase']
        self.min_loss = checkpoint['min_loss']
        
        # Load model version if available in checkpoint (for backwards compatibility)
        if 'model_version' in checkpoint:
            checkpoint_version = checkpoint['model_version']
            if checkpoint_version != self.model_version:
                self.logger.warning(f"Checkpoint model version ({checkpoint_version}) differs from current ({self.model_version})")
        
        # Update dataset training phase
        self.dataset.set_training_phase(self.training_phase)
        
        self.logger.info(f"Resuming from epoch {self.start_epoch}")
        self.logger.info(f"Training phase: {self.training_phase}")
        self.logger.info(f"Min loss so far: {self.min_loss:.4f}")
    
    def set_training_phase(self, phase):
        """
        Switch to a different training phase and update optimizer parameters.
        
        This method updates the learning rate and weight decay of the existing optimizer
        rather than reinitializing it, which preserves Adam's momentum and variance estimates.
        
        Args:
            phase: New phase (1, 2, 3 or 4)
        """
        self.training_phase = phase
        self.dataset.set_training_phase(phase)
        
        # Get phase-specific optimizer parameters
        lr, weight_decay = util.get_phase_optimizer_params(phase)
        
        # Update optimizer parameters without reinitializing (preserves Adam state)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['weight_decay'] = weight_decay
        
        self.logger.info(f"Switched to training phase {phase} (LR={lr}, Weight Decay={weight_decay})")

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
                if self.model_version == 'v2':
                    average_el_loss = total_el_loss / (batch_idx + 1)
                    self.logger.info(f"| Epoch {epoch} | Average Loss: {average_loss:.4f} | Average AZ Loss: {average_az_loss:.4f} | Average EL Loss: {average_el_loss:.4f} |")
                else:
                    self.logger.info(f"| Epoch {epoch} | Average Loss: {average_loss:.4f} | Average AZ Loss: {average_az_loss:.4f} |")

            audio = audio.to(self.device)
            vad = vad.to(self.device)
            mic_coordinates = room_params['mic_coords'].to(self.device)
            source_coordinates = room_params['source_coords'].to(self.device)

            self.optimizer.zero_grad()

            # Version-specific forward pass
            if self.model_version == 'v1':
                # V1: Single output (azimuth only)
                x_out, target, vad_framed = self.model(audio, mic_coordinates, vad, source_coordinates, return_target=True)
                loss = util.compute_bce_loss(epoch, x_out, target, vad_framed)
                loss_az = loss  # For logging consistency
            else:
                # V2: Two outputs (azimuth + elevation)
                x_out_az, x_out_el, target_az, target_el, vad_framed = self.model(audio, mic_coordinates, vad, source_coordinates, return_target=True)
                loss_az = util.compute_bce_loss(epoch, x_out_az, target_az, vad_framed)
                loss_el = util.compute_bce_loss(epoch, x_out_el, target_el, vad_framed)
                loss = loss_az + loss_el
                total_el_loss += loss_el.item()
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()

            total_loss += loss.item()
            total_az_loss += loss_az.item()

        avg_total_loss = total_loss / len(dataloader)
        avg_az_loss = total_az_loss / len(dataloader)
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/Total', avg_total_loss, epoch)
        self.writer.add_scalar('Loss/Azimuth', avg_az_loss, epoch)
        
        if self.model_version == 'v2':
            avg_el_loss = total_el_loss / len(dataloader)
            self.writer.add_scalar('Loss/Elevation', avg_el_loss, epoch)
        
        return avg_total_loss
    


    def __call__(self):
        # Log comprehensive training configuration
        initial_lr, initial_wd = util.get_phase_optimizer_params(self.training_phase)
        config = {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'mpe_type': self.MPE_type,
            'device': self.device,
            'data_path': self.data_path,
            'mode': self.mode,
            'max_microphones': self.max_microphones,
            'sample_rate': self.sample_rate,
            'noise_probability': self.noise_probability,
            'training_phase': self.training_phase,
            'total_samples': len(self.dataset),
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'initial_lr': initial_lr,
            'initial_wd': initial_wd
        }
        util.log_training_config(self.logger, config)
        
        self.logger.info(f"Starting training of {self.model_name} on {self.device}...")

        model_dir = os.path.join(os.getcwd(), "saved_models", self.model_name)
        os.makedirs(model_dir, exist_ok=True)
        checkpoint_path = os.path.join(model_dir, f"{self.model_name}.tar")

        for epoch in range(self.start_epoch, self.epochs):
            # Automatic phase switching based on epoch
            new_phase = util.should_switch_phase(epoch, self.training_phase)
            if new_phase is not None:
                self.logger.info(f"{'='*50}")
                self.logger.info(f"Auto-switching to Phase {new_phase} at epoch {epoch}")
                self.logger.info(f"{'='*50}\n")
                self.set_training_phase(new_phase)
            
            # Update gamma values based on epoch
            gamma = util.get_gamma_for_epoch(epoch)
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
                    'model_version': self.model_version,
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tune GI-DOAEnet model')
    
    # Model configuration
    parser.add_argument('--model_name', type=str, default='v2',
                        help='Name of the model (default: GI_DOAEnet_fine_tuned)')
    parser.add_argument('--model_version', type=str, default='v2', choices=['v1', 'v2'],
                        help='Model version: v1 (azimuth only) or v2 (azimuth + elevation) (default: v1)')
    parser.add_argument('--mpe_type', type=str, default='PM', choices=['PM', 'FM'],
                        help='Microphone positional encoding type (default: PM)')
    
    # Data configuration
    parser.add_argument('--data_path', type=str, default=os.path.join(os.getcwd(), "data"),
                        help='Path to training data (default: ./data)')
    parser.add_argument('--mode', type=str, default='validation',
                        help='Dataset mode (default: validation)')
    parser.add_argument('--max_microphones', type=int, default=12,
                        help='Maximum number of microphones (default: 12)')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate in Hz (default: 16000)')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--epochs', type=int, default=350,
                        help='Total number of training epochs (default: 300)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Starting epoch (optional, for manual epoch override)')
    parser.add_argument('--noise_probability', type=float, default=1.0,
                        help='Probability of adding noise during training (default: 1.0)')
    
    # Checkpoint configuration
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint if available')
    parser.add_argument('--checkpoint_name', type=str, default=None,
                        help='Name of the checkpoint to resume from (optional)')

    
    # Device configuration
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for training (default: auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    trainer = Trainer(
        model_name=args.model_name,
        data_path=args.data_path,
        mode=args.mode,
        device=device,
        max_microphones=args.max_microphones,
        sample_rate=args.sample_rate,
        MPE_type=args.mpe_type,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        start_epoch=args.start_epoch,
        noise_probability=args.noise_probability,
        model_version=args.model_version,
        resume_from_checkpoint=args.resume,
        checkpoint_name=args.checkpoint_name,
    )
    
    trainer()
    
    # Close TensorBoard writer
    trainer.writer.close()
