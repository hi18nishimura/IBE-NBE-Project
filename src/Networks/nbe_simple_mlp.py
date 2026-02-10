import torch
import torch.nn as nn

class NbeSimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # Hidden Layer 1: Input -> 64
        self.fc1 = nn.Linear(input_dim, 64)
        
        # Hidden Layer 2: 64 -> 32
        self.fc2 = nn.Linear(64, 32)
        
        # Hidden Layer 3: 32 -> 16
        self.fc3 = nn.Linear(32, 16)
        
        # Output Layer: 16 -> Output
        self.fc4 = nn.Linear(16, output_dim)
        
        # Activation function for hidden layers
        self.tanh = nn.Tanh()
        
        # Final activation function base
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.tanh(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.tanh(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.tanh(x)
        
        # Layer 4 (Output)
        x = self.fc4(x)
        
        # Scale sigmoid to 0.1 - 0.9 range
        # 0.1 + (0.9 - 0.1) * sigmoid(x) = 0.1 + 0.8 * sigmoid(x)
        x = 0.1 + 0.8 * self.sigmoid(x)
        
        return x

class NbeSimpleSplitMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.output_dim = output_dim
        
        # Hidden Layer 1: Input -> 64
        self.fc1 = nn.Linear(input_dim, 64)
        
        # Hidden Layer 2: 64 -> 32
        self.fc2 = nn.Linear(64, 32)
        
        # Hidden Layer 3: 32 -> 16
        self.fc3 = nn.Linear(32, 16)
        
        # Output Layer: Split based on output_dim
        if output_dim == 9:
            # Split into 3 parts (3, 3, 3)
            self.fc4_1 = nn.Linear(16, 3)
            self.fc4_2 = nn.Linear(16, 3)
            self.fc4_3 = nn.Linear(16, 3)
        elif output_dim == 6:
             # Split into 2 parts (3, 3)
            self.fc4_1 = nn.Linear(16, 3)
            self.fc4_2 = nn.Linear(16, 3)
        else:
             raise ValueError(f"NbeSimpleSplitMLP only supports output_dim of 6 or 9, but got {output_dim}")
        
        # Activation function for hidden layers
        self.tanh = nn.Tanh()
        
        # Final activation function base
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.tanh(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.tanh(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.tanh(x)
        
        # Layer 4 (Output Split)
        if self.output_dim == 9:
            out1 = self.fc4_1(x)
            out2 = self.fc4_2(x)
            out3 = self.fc4_3(x)
            
            out1 = 0.1 + 0.8 * self.sigmoid(out1)
            out2 = 0.1 + 0.8 * self.sigmoid(out2)
            out3 = 0.1 + 0.8 * self.sigmoid(out3)
            
            x = torch.cat([out1, out2, out3], dim=-1)
            
        elif self.output_dim == 6:
            out1 = self.fc4_1(x)
            out2 = self.fc4_2(x)
            
            out1 = 0.1 + 0.8 * self.sigmoid(out1)
            out2 = 0.1 + 0.8 * self.sigmoid(out2)
            
            x = torch.cat([out1, out2], dim=-1)
        
        return x

class NbeAutoEncoderMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # AutoEncoder Encoder
        # Input -> 64
        self.ae_enc1 = nn.Linear(input_dim, 64)
        # 64 -> 32
        self.ae_enc2 = nn.Linear(64, 32)
        
        # AutoEncoder Decoder
        # 32 -> 64
        self.ae_dec1 = nn.Linear(32, 64)
        # 64 -> Input
        self.ae_dec2 = nn.Linear(64, input_dim)
        
        # MLP Part (Same as NbeSimpleMLP)
        # Hidden Layer 1: Input -> 64
        self.fc1 = nn.Linear(input_dim, 64)
        
        # Hidden Layer 2: 64 -> 32
        self.fc2 = nn.Linear(64, 32)
        
        # Hidden Layer 3: 32 -> 16
        self.fc3 = nn.Linear(32, 16)
        
        # Output Layer: 16 -> Output
        self.fc4 = nn.Linear(16, output_dim)
        
        # Activation function for hidden layers
        self.tanh = nn.Tanh()
        
        # Final activation function base
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # --- AutoEncoder Forward ---
        # Encoder
        enc = self.ae_enc1(x)
        enc = self.tanh(enc)
        enc = self.ae_enc2(enc)
        enc = self.tanh(enc)
        
        # Decoder
        dec = self.ae_dec1(enc)
        dec = self.tanh(dec)
        rec_x = 0.1 + 0.8 * self.sigmoid(self.ae_dec2(dec))
        # Use simple linear output for reconstruction to match input scale freely
        
        # --- MLP Forward ---
        # Use reconstructed input for MLP
        h = self.fc1(rec_x)
        h = self.tanh(h)
        
        h = self.fc2(h)
        h = self.tanh(h)
        
        h = self.fc3(h)
        h = self.tanh(h)
        
        out = self.fc4(h)
        
        # Scale sigmoid to 0.1 - 0.9 range
        out = 0.1 + 0.8 * self.sigmoid(out)
        
        return out, rec_x

