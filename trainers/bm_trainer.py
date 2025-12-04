"""
Trainer for Boltzmann Machine models.
"""

import os
import torch
import numpy as np
from typing import Dict, Any, Optional, Literal
from torch.utils.data import DataLoader
from pathlib import Path

from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine as GRBM
from dwave.samplers import SimulatedAnnealingSampler


class BoltzmannMachineTrainer:
    """
    Trainer for Boltzmann Machine with train/val/test support.
    """

    def __init__(
        self,
        model: GRBM,
        config: Dict[str, Any],
        device: torch.device,
        sampler: Optional[Any] = None
    ):
        """
        Initialize the trainer.

        Args:
            model: GraphRestrictedBoltzmannMachine instance
            config: Training configuration dictionary
            device: torch device (CPU or CUDA)
            sampler: D-Wave sampler (default: SimulatedAnnealingSampler)
        """
        self.device = device
        self.model = model.to(device)  # Move model to device
        self.config = config
        self.training_config = config['training']

        # Sampler
        self.sampler = sampler if sampler is not None else SimulatedAnnealingSampler()

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Training state
        self.current_epoch = 0
        self.patience_counter = 0

        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'grad_norm': [],
            'beta': []
        }

        # Early stopping configuration
        self.early_stopping_config = self.training_config.get('early_stopping', {})
        self.early_stopping_enabled = self.early_stopping_config.get('enabled', False)
        self.patience = self.early_stopping_config.get('patience', 20)
        self.min_delta = self.early_stopping_config.get('min_delta', 1e-4)
        self.monitor_metric = self.early_stopping_config.get('metric', 'val_loss')
        self.monitor_mode = self.early_stopping_config.get('mode', 'min')
        self.restore_best = self.early_stopping_config.get('restore_best_weights', True)

        # Best metric value (initialize based on mode)
        if self.monitor_mode == 'min':
            self.best_metric_value = float('inf')
        else:
            self.best_metric_value = float('-inf')

        # Store best model state
        self.best_model_state = None

        # Checkpoint directory
        self.checkpoint_dir = self.training_config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Gradient clipping configuration
        self.grad_clip_config = self.training_config.get('gradient_clipping', {})
        self.grad_clip_enabled = self.grad_clip_config.get('enabled', False)
        self.grad_clip_method = self.grad_clip_config.get('method', 'norm')
        self.max_grad_norm = self.grad_clip_config.get('max_norm', 1.0)
        self.max_grad_value = self.grad_clip_config.get('max_value', 0.5)

        # Regularization configuration
        self.reg_config = self.training_config.get('regularization', {})
        self.linear_l2 = self.reg_config.get('linear_l2', 0.0)
        self.quadratic_l2 = self.reg_config.get('quadratic_l2', 0.0)
        self.quadratic_l1 = self.reg_config.get('quadratic_l1', 0.0)

        # Learning rate scheduler
        self.lr_scheduler = self._create_lr_scheduler()

    def _create_optimizer(self):
        """Create optimizer based on config."""
        optimizer_name = self.training_config['optimizer'].lower()
        lr = self.training_config['learning_rate']

        # Get optimizer parameters
        optimizer_params = self.training_config.get('optimizer_params', {})

        if optimizer_name == 'sgd':
            momentum = optimizer_params.get('momentum', 0.0)
            dampening = optimizer_params.get('dampening', 0.0)
            nesterov = optimizer_params.get('nesterov', False)
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                dampening=dampening,
                nesterov=nesterov
            )
        elif optimizer_name == 'adam':
            betas = tuple(optimizer_params.get('betas', [0.9, 0.999]))
            eps = optimizer_params.get('eps', 1e-8)
            weight_decay = optimizer_params.get('weight_decay', 0.0)
            amsgrad = optimizer_params.get('amsgrad', False)
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsgrad
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _create_lr_scheduler(self):
        """Create learning rate scheduler based on config."""
        lr_config = self.training_config.get('lr_scheduler', {})

        if not lr_config.get('enabled', False):
            return None

        scheduler_type = lr_config.get('type', 'plateau').lower()

        if scheduler_type == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=lr_config.get('factor', 0.5),
                patience=lr_config.get('patience', 10),
                min_lr=lr_config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=lr_config.get('step_size', 100),
                gamma=lr_config.get('gamma', 0.5)
            )
        elif scheduler_type == 'exponential':
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=lr_config.get('gamma', 0.95)
            )
        elif scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=lr_config.get('T_max', 50),
                eta_min=lr_config.get('eta_min', 1e-6)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def _sample_from_model(self) -> torch.Tensor:
        """Sample from the current model."""
        model_sample_size = self.training_config['model_sample_size']
        prefactor = self.training_config['prefactor']

        samples = self.model.sample(
            self.sampler,
            prefactor=prefactor,
            sample_params={
                'num_reads': model_sample_size,
                'beta_range': [1.0, 1.0],
                'proposal_acceptance_criteria': 'Gibbs'
            },
            as_tensor=True
        )

        return samples

    def _compute_loss(
        self,
        data_batch: torch.Tensor,
        model_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the quasi-objective loss with regularization.

        Args:
            data_batch: Batch of observed data, shape (batch_size, n_visible)
            model_samples: Samples from model, shape (n_samples, n_total)

        Returns:
            Loss tensor
        """
        hidden_kind = self.training_config.get('hidden_kind')
        prefactor = self.training_config['prefactor']

        # Compute base loss
        loss = self.model.quasi_objective(
            data_batch,
            model_samples,
            kind=hidden_kind,
            prefactor=prefactor if hidden_kind is not None else None,
            sampler=self.sampler if hidden_kind == "sampling" else None,
            sample_kwargs={'num_reads': 100} if hidden_kind == "sampling" else None
        )

        # Add regularization
        if self.linear_l2 > 0:
            loss = loss + self.linear_l2 * self.model._linear.pow(2).sum()

        if self.quadratic_l2 > 0:
            loss = loss + self.quadratic_l2 * self.model._quadratic.pow(2).sum()

        if self.quadratic_l1 > 0:
            loss = loss + self.quadratic_l1 * self.model._quadratic.abs().sum()

        return loss

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()

        epoch_losses = []
        epoch_grad_norms = []

        # Sample once from model for the entire epoch (more efficient)
        model_samples = self._sample_from_model()

        for batch_idx, data_batch in enumerate(train_loader):
            # Move data to device
            data_batch = data_batch.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Compute loss
            loss = self._compute_loss(data_batch, model_samples)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.grad_clip_enabled:
                if self.grad_clip_method == 'norm':
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                elif self.grad_clip_method == 'value':
                    torch.nn.utils.clip_grad_value_(
                        self.model.parameters(),
                        self.max_grad_value
                    )

            # Compute gradient norm (after clipping)
            grad_norm = (
                self.model._linear.grad.abs().mean().item() +
                self.model._quadratic.grad.abs().mean().item()
            ) / 2

            # Update parameters
            self.optimizer.step()

            # Record metrics
            epoch_losses.append(loss.item())
            epoch_grad_norms.append(grad_norm)

        # Estimate beta
        try:
            beta = self.model.estimate_beta(model_samples)
        except:
            beta = None

        metrics = {
            'loss': np.mean(epoch_losses),
            'grad_norm': np.mean(epoch_grad_norms),
            'beta': beta
        }

        return metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        val_losses = []

        # Sample from model
        model_samples = self._sample_from_model()

        with torch.no_grad():
            for data_batch in val_loader:
                # Move data to device
                data_batch = data_batch.to(self.device)
                loss = self._compute_loss(data_batch, model_samples)
                val_losses.append(loss.item())

        metrics = {
            'val_loss': np.mean(val_losses)
        }

        return metrics

    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Test the model.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary with test metrics
        """
        self.model.eval()

        test_losses = []

        # Sample from model
        model_samples = self._sample_from_model()

        with torch.no_grad():
            for data_batch in test_loader:
                # Move data to device
                data_batch = data_batch.to(self.device)
                loss = self._compute_loss(data_batch, model_samples)
                test_losses.append(loss.item())

        metrics = {
            'test_loss': np.mean(test_losses)
        }

        return metrics

    def check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """
        Check if training should stop early based on monitored metric.

        Args:
            metrics: Dictionary containing current metrics

        Returns:
            True if training should stop
        """
        if not self.early_stopping_enabled:
            return False

        # Get the metric value to monitor
        if self.monitor_metric not in metrics:
            return False

        current_value = metrics[self.monitor_metric]

        # Check if this is an improvement
        is_improvement = False
        if self.monitor_mode == 'min':
            if current_value < self.best_metric_value - self.min_delta:
                is_improvement = True
        else:  # mode == 'max'
            if current_value > self.best_metric_value + self.min_delta:
                is_improvement = True

        if is_improvement:
            self.best_metric_value = current_value
            self.patience_counter = 0

            # Save best model state for potential restoration
            if self.restore_best:
                self.best_model_state = {
                    'linear': self.model.linear.detach().clone(),
                    'quadratic': self.model.quadratic.detach().clone()
                }
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered:")
                print(f"  Monitored metric: {self.monitor_metric}")
                print(f"  Mode: {self.monitor_mode}")
                print(f"  Best value: {self.best_metric_value:.4f}")
                print(f"  Patience: {self.patience} epochs")

                # Restore best weights if requested
                if self.restore_best and self.best_model_state is not None:
                    print(f"  Restoring best model weights...")
                    self.model._linear.data = self.best_model_state['linear']
                    self.model._quadratic.data = self.best_model_state['quadratic']

                return True
            return False

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': {
                'linear': self.model.linear.detach().cpu(),
                'quadratic': self.model.quadratic.detach().cpu()
            },
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_metric_value': self.best_metric_value,
            'monitor_metric': self.monitor_metric
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath)
        self.model._linear.data = checkpoint['model_state_dict']['linear']
        self.model._quadratic.data = checkpoint['model_state_dict']['quadratic']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.history = checkpoint['history']
        self.best_metric_value = checkpoint.get('best_metric_value', float('inf'))
        self.monitor_metric = checkpoint.get('monitor_metric', 'val_loss')

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True
    ):
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            verbose: Print progress
        """
        n_epochs = self.training_config['n_epochs']
        log_interval = self.config.get('logging', {}).get('log_interval', 10)

        if verbose:
            print(f"\n{'='*70}")
            print("TRAINING BOLTZMANN MACHINE")
            print(f"{'='*70}")
            print(f"Epochs:          {n_epochs}")
            print(f"Batch size:      {self.training_config['batch_size']}")
            print(f"Learning rate:   {self.training_config['learning_rate']}")
            print(f"Optimizer:       {self.training_config['optimizer']}")
            print(f"Model samples:   {self.training_config['model_sample_size']}")

            # Gradient clipping
            if self.grad_clip_enabled:
                print(f"\nGradient Clipping:")
                print(f"  Method:        {self.grad_clip_method}")
                if self.grad_clip_method == 'norm':
                    print(f"  Max norm:      {self.max_grad_norm}")
                else:
                    print(f"  Max value:     {self.max_grad_value}")

            # Regularization
            if self.linear_l2 > 0 or self.quadratic_l2 > 0 or self.quadratic_l1 > 0:
                print(f"\nRegularization:")
                if self.linear_l2 > 0:
                    print(f"  Linear L2:     {self.linear_l2}")
                if self.quadratic_l2 > 0:
                    print(f"  Quadratic L2:  {self.quadratic_l2}")
                if self.quadratic_l1 > 0:
                    print(f"  Quadratic L1:  {self.quadratic_l1}")

            # LR Scheduler
            if self.lr_scheduler is not None:
                print(f"\nLR Scheduler:")
                print(f"  Type:          {self.training_config['lr_scheduler']['type']}")
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    print(f"  Factor:        {self.training_config['lr_scheduler']['factor']}")
                    print(f"  Patience:      {self.training_config['lr_scheduler']['patience']}")

            # Early stopping
            if self.early_stopping_enabled:
                print(f"\nEarly Stopping:")
                print(f"  Metric:        {self.monitor_metric}")
                print(f"  Mode:          {self.monitor_mode}")
                print(f"  Patience:      {self.patience}")
                print(f"  Min delta:     {self.min_delta}")
                print(f"  Restore best:  {self.restore_best}")
            print(f"{'='*70}\n")

        for epoch in range(n_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics['val_loss']
            else:
                val_metrics = {}
                val_loss = None

            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['grad_norm'].append(train_metrics['grad_norm'])
            if train_metrics['beta'] is not None:
                self.history['beta'].append(train_metrics['beta'])
            if val_loss is not None:
                self.history['val_loss'].append(val_loss)

            # Update learning rate scheduler
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # ReduceLROnPlateau needs a metric
                    metric_to_monitor = val_loss if val_loss is not None else train_metrics['loss']
                    self.lr_scheduler.step(metric_to_monitor)
                else:
                    # Other schedulers don't need a metric
                    self.lr_scheduler.step()

            # Log progress
            if verbose and epoch % log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                log_str = f"Epoch {epoch:3d}: Train Loss={train_metrics['loss']:8.4f}"
                if val_loss is not None:
                    log_str += f", Val Loss={val_loss:8.4f}"
                log_str += f", Grad={train_metrics['grad_norm']:.4f}"
                log_str += f", LR={current_lr:.2e}"
                if train_metrics['beta'] is not None:
                    log_str += f", beta={train_metrics['beta']:.4f}"
                print(log_str)

            # Combine all metrics for early stopping check
            all_metrics = {
                'train_loss': train_metrics['loss'],
                'grad_norm': train_metrics['grad_norm']
            }
            if val_loss is not None:
                all_metrics['val_loss'] = val_loss

            # Save best model based on monitored metric
            if self.training_config.get('save_best_model', False):
                # Check if current is best based on monitor metric
                current_metric = all_metrics.get(self.monitor_metric)
                if current_metric is not None:
                    is_best = False
                    if self.monitor_mode == 'min':
                        if current_metric < self.best_metric_value:
                            is_best = True
                    else:
                        if current_metric > self.best_metric_value:
                            is_best = True

                    if is_best:
                        best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                        self.save_checkpoint(best_model_path)
                        if verbose and epoch % log_interval == 0:
                            print(f"  -> Saved best model ({self.monitor_metric}={current_metric:.4f})")

            # Early stopping
            if self.check_early_stopping(all_metrics):
                break

        if verbose:
            print(f"\n{'='*70}")
            print("TRAINING COMPLETE")
            print(f"{'='*70}")
            print(f"Final train loss: {self.history['train_loss'][-1]:.4f}")
            if len(self.history['val_loss']) > 0:
                print(f"Final val loss:   {self.history['val_loss'][-1]:.4f}")
            if self.early_stopping_enabled:
                print(f"Best {self.monitor_metric}: {self.best_metric_value:.4f}")
            print(f"{'='*70}\n")

    def get_history(self) -> Dict[str, list]:
        """Get training history."""
        return self.history
