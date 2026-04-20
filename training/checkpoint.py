"""
Cluster-Scale Checkpoint Resumption — fast-gpt-lab
Fault-tolerant checkpoint saving and loading for interrupted training runs.
"""
import os
import glob
import torch
import shutil

class CheckpointManager:
    """
    Manages N most recent checkpoints to prevent disk overflow while
    allowing resumption from hardware failures.
    """
    def __init__(self, checkpoint_dir: str, keep_last_n: int = 3):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self, state_dict: dict, step: int, is_best: bool = False):
        """Save a new checkpoint and prune old ones."""
        filename = f"ckpt_step_{step:08d}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # Save temporary then atomic rename for crash safety
        tmp_path = filepath + ".tmp"
        torch.save(state_dict, tmp_path)
        os.replace(tmp_path, filepath)
        print(f"💾 Checkpoint saved at step {step}")

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "ckpt_best.pt")
            shutil.copyfile(filepath, best_path)

        self._prune_old_checkpoints()

    def _prune_old_checkpoints(self):
        """Keep only the N most recent step checkpoints."""
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "ckpt_step_*.pt"))
        checkpoints.sort(key=os.path.getmtime)
        
        if len(checkpoints) > self.keep_last_n:
            for ckpt in checkpoints[:-self.keep_last_n]:
                try:
                    os.remove(ckpt)
                except OSError:
                    pass

    def load_latest(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None) -> int:
        """Attempt to load the most recent checkpoint. Returns the step number."""
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "ckpt_step_*.pt"))
        if not checkpoints:
            return 0
            
        latest_ckpt = max(checkpoints, key=os.path.getmtime)
        print(f"🔄 Resuming from checkpoint: {latest_ckpt}")
        
        state = torch.load(latest_ckpt, map_location="cpu")
        model.load_state_dict(state.get("model_state", state))
        
        if optimizer and "optimizer_state" in state:
            optimizer.load_state_dict(state["optimizer_state"])
            
        return state.get("step", 0)
