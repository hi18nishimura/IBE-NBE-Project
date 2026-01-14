import torch
import torch.nn as nn

class NbeDenoiseAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, bottleneck_dim=64, num_layers=2):
        super().__init__()
        
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        # Encoder
        encoder_layers = []
        if num_layers == 1:
            encoder_layers.extend([
                nn.Linear(input_dim, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                self.activation
            ])
        else:
            # Input -> Hidden
            encoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation
            ])
            
            # Hidden -> Hidden
            for _ in range(num_layers - 2):
                encoder_layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    self.activation
                ])
                
            # Hidden -> Bottleneck
            encoder_layers.extend([
                nn.Linear(hidden_dim, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                self.activation
            ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        if num_layers == 1:
            decoder_layers.append(nn.Linear(bottleneck_dim, input_dim))
        else:
            # Bottleneck -> Hidden
            decoder_layers.extend([
                nn.Linear(bottleneck_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation
            ])
            
            # Hidden -> Hidden
            for _ in range(num_layers - 2):
                decoder_layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    self.activation
                ])
                
            # Hidden -> Input
            decoder_layers.append(nn.Linear(hidden_dim, input_dim))
            
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.sigmoid = nn.Sigmoid()
        
        self._initialize_weights()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return self.sigmoid(decoded) * 0.8 + 0.1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
