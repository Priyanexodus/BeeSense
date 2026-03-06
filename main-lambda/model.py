import torch
import torch.nn as nn

class HybridBeeModel(nn.Module):
    """Neural network for bee health classification using audio and telemetry data."""
    
    def __init__(self, num_classes=2):
        super(HybridBeeModel, self).__init__()
        
        # Audio CNN branch
        self.audio_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        self.audio_flat_size = 32 * 16 * 54

        # Telemetry branch (8 features)
        self.telemetry_fc = nn.Sequential(
            nn.Linear(8, 16), 
            nn.ReLU()
        )

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.audio_flat_size + 16, 128), 
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, audio, telemetry):
        """
        Forward pass through the network.
        
        Args:
            audio: Audio spectrogram tensor [batch, 1, 64, 216]
            telemetry: Telemetry features tensor [batch, 8]
        
        Returns:
            Class logits [batch, num_classes]
        """
        x1 = self.audio_cnn(audio)
        x2 = self.telemetry_fc(telemetry)
        return self.classifier(torch.cat((x1, x2), dim=1))