"""
Enterprise Telemetry Integration — fast-gpt-lab
Weights & Biases integration for cluster-scale experiment tracking.
"""
import os
import torch
from typing import Dict, Any, Optional

class TelemetryManager:
    """
    Manages experiment tracking, metrics logging, and artifact saving via W&B.
    Fails gracefully if W&B is not installed or offline.
    """
    def __init__(self, project_name: str, run_name: str, config: Dict[str, Any], enabled: bool = True):
        self.enabled = enabled
        self.step = 0
        
        if self.enabled:
            try:
                import wandb
                self.wandb = wandb
                # Initialize W&B run
                self.run = self.wandb.init(
                    project=project_name,
                    name=run_name,
                    config=config,
                    resume="allow"
                )
                print(f"🚀 W&B Telemetry initialized. Run tracking at: {self.run.url}")
            except ImportError:
                print("⚠️ W&B not installed. Run `pip install wandb` for telemetry. Disabling tracking.")
                self.enabled = False

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log a dictionary of metrics."""
        if not self.enabled:
            return
            
        current_step = step if step is not None else self.step
        self.wandb.log(metrics, step=current_step)
        self.step = current_step + 1

    def watch_model(self, model: torch.nn.Module, log_freq: int = 100):
        """Hook into model to track gradients and topology."""
        if not self.enabled:
            return
        self.wandb.watch(model, log="all", log_freq=log_freq)

    def save_artifact(self, file_path: str, artifact_type: str = "model", name: str = "checkpoint"):
        """Upload a file as a tracked artifact."""
        if not self.enabled:
            return
        if os.path.exists(file_path):
            artifact = self.wandb.Artifact(name, type=artifact_type)
            artifact.add_file(file_path)
            self.run.log_artifact(artifact)

    def finish(self):
        """Close the W&B run cleanly."""
        if self.enabled:
            self.wandb.finish()
