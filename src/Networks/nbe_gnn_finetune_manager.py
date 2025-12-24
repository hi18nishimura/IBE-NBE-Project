import os
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from omegaconf import OmegaConf
import sys

# Add workspace root to sys.path if needed
sys.path.append("/workspace")

from src.Networks.nbe_gnn import NbeGNN

class NbeGNNFinetuneManager:
    def __init__(self, model_dir: str, node_ids: List[int], device: str = 'cpu'):
        """
        Args:
            model_dir: Directory containing model subdirectories (e.g. /workspace/NBE/GNN_Global_fc_numlayer3_weight)
            node_ids: List of node IDs to load
            device: 'cpu' or 'cuda'
        """
        self.model_dir = Path(model_dir)
        self.node_ids = node_ids
        self.device = device
        self.models: Dict[int, NbeGNN] = {}
        self.configs: Dict[int, Any] = {}
        
        self._load_models()

    def _load_models(self):
        for node_id in self.node_ids:
            node_dir = self.model_dir / str(node_id)
            finetune_path = node_dir / "best_finetune.pth"
            pretrain_path = node_dir / "best_pretrain.pth"
            
            if finetune_path.exists():
                model_path = finetune_path
            elif pretrain_path.exists():
                model_path = pretrain_path
            else:
                raise FileNotFoundError(f"Neither best_finetune.pth nor best_pretrain.pth found for node {node_id} in {node_dir}")
                
            try:
                self._load_single_model(node_id, model_path)
            except Exception as e:
                print(f"Error loading model for node {node_id}: {e}")

    def _load_single_model(self, node_id: int, model_path: Path):
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Extract and process config
        cfg = checkpoint.get('cfg', {})
        
        # Handle OmegaConf
        if OmegaConf.is_config(cfg):
            cfg = OmegaConf.to_container(cfg, resolve=True)
        
        # Handle serialized OmegaConf as dict (with _content)
        if isinstance(cfg, dict) and '_content' in cfg:
            cfg = cfg['_content']
            
        # Try to clean up potential OmegaConf objects inside the dict
        try:
            cfg = OmegaConf.create(cfg)
            cfg = OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            pass
            
        self.configs[node_id] = cfg
        
        state_dict = checkpoint.get('model_state', checkpoint.get('model_state_dict'))
        if state_dict is None:
            raise ValueError(f"No model state found in checkpoint for node {node_id}")

        # Extract hyperparameters from config
        hidden_size = cfg.get('hidden_size', 64)
        num_layers = cfg.get('num_layers', 2)
        dropout = cfg.get('dropout', 0.5)
        gnn_type = cfg.get('gnn_type', 'GCN')
        multi_fully_layer = cfg.get('multi_fully_layer', 1)
        
        # Infer input_size from state_dict (first layer weights)
        input_size = self._infer_input_size(state_dict, gnn_type)
        if input_size is None:
            raise ValueError(f"Could not infer input_size from state_dict for node {node_id}")

        # Infer output_size from config or state_dict
        output_size = cfg.get('output_size')
        if output_size is None:
            output_size = self._infer_output_size(state_dict)
            
        if output_size is None:
             raise ValueError(f"Could not infer output_size for node {node_id}")

        # Instantiate model
        # Note: Assuming return_only_central=True as this is likely for node-specific tasks
        model = NbeGNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            dropout=dropout,
            gnn_type=gnn_type,
            finetune=True,
            return_only_central=True,
            multi_fully_layer=multi_fully_layer
        )
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        # model.eval() # Keep in train mode or eval mode? Usually eval for loading. User can switch.
        
        self.models[node_id] = model
        print(f"Loaded model for node {node_id}: Input={input_size}, Output={output_size}, Hidden={hidden_size}, Layers={num_layers}, Type={gnn_type}")

    def _infer_input_size(self, state_dict: Dict[str, Any], gnn_type: str) -> Optional[int]:
        # Try to find the first layer's weight
        # NbeGNN uses self.convs which is a ModuleList
        # The first layer is convs.0
        
        # For GCNConv, the weight parameter is 'lin.weight' (if cached=False, etc.) or similar depending on version
        # In PyG 2.x GCNConv: lin is a Linear layer. weight shape is (out, in).
        
        keys_to_check = [
            'convs.0.lin.weight',      # GCN
            'convs.0.lin_src.weight',  # GAT
            'convs.0.weight'           # Generic fallback
        ]
        
        for key in keys_to_check:
            if key in state_dict:
                return state_dict[key].shape[1]
        
        # Fallback: search for any convs.0 weight that looks like a matrix
        for key, tensor in state_dict.items():
            if key.startswith('convs.0.') and 'weight' in key and tensor.dim() == 2:
                # Assuming (out, in)
                return tensor.shape[1]
                
        return None

    def _infer_output_size(self, state_dict: Dict[str, Any]) -> Optional[int]:
        # Check readout layer
        if 'readout.weight' in state_dict:
            return state_dict['readout.weight'].shape[0]
        if 'readout.bias' in state_dict:
            return state_dict['readout.bias'].shape[0]
        return None

    def get_models(self) -> Dict[int, NbeGNN]:
        return self.models

    def train(self):
        for model in self.models.values():
            model.train()

    def eval(self):
        for model in self.models.values():
            model.eval()

    def forward(self, x_dict: Dict[int, torch.Tensor], edge_index_dict: Dict[int, torch.Tensor], global_features_dict: Optional[Dict[int, torch.Tensor]] = None) -> Dict[int, torch.Tensor]:
        """
        Forward pass for all models.
        Args:
            x_dict: {node_id: input_tensor}
            edge_index_dict: {node_id: edge_index_tensor}
            global_features_dict: {node_id: global_features_tensor} (Optional)
        Returns:
            {node_id: output_tensor}
        """
        outputs = {}
        for node_id, x in x_dict.items():
            if node_id in self.models and node_id in edge_index_dict:
                model = self.models[node_id]
                edge_index = edge_index_dict[node_id]
                
                x = x.to(self.device)
                edge_index = edge_index.to(self.device)
                
                gf = None
                if global_features_dict and node_id in global_features_dict:
                    gf = global_features_dict[node_id].to(self.device)
                    
                outputs[node_id] = model(x, edge_index, global_features=gf)
        return outputs
