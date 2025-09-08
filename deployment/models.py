import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models


class TextEnconder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', use_auth_token=False)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.projection = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask):
        # Extract BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # USE [CLS] token representation
        pooler_output = outputs.pooler_output
        
        return self.projection(pooler_output)
    

class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vision_models.video.r3d_18(pretrained=True)
        
        # Freeze ResNet layers
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # num_fts --> number of features in the last fully connected layer
        num_fts = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_fts, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # Change shape from (B, T, C, H, W) to (B*T, C, H, W)
        return self.backbone(x)
    

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(64,64,kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64,128,kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        for param in self.conv_layers.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )       

    def forward(self, x):
        x = x.squeeze(1)  # Remove channel dimension if present

        features = self.conv_layers(x)
        # Features output shape: (Batch_size, 128, 1)

        return self.projection(features.squeeze(-1))  # Remove the last dimension
    

class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()

        #Encoders
        self.text_encoder = TextEnconder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Classification heads
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7)  # 7 emotion classes
        )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)  # 3 sentiment classes
        )

    def forward(self, text_inputs, video_frames, audio_features):
        # Encode text, video, and audio
        text_features = self.text_encoder(text_inputs['input_ids'], text_inputs['attention_mask'])
        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)

        # Concatenate features
        combined_features = torch.cat((text_features, video_features, audio_features), dim=1)
        # [batch_size, 384] where 128 (text) + 128 (video) + 128 (audio)

        # Apply fusion layer
        fused_features = self.fusion_layer(combined_features)

        # Classify emotions and sentiments
        emotion_output = self.emotion_classifier(fused_features)
        sentiment_output = self.sentiment_classifier(fused_features)

        return {
            'emotion_output': emotion_output,
            'sentiment_output': sentiment_output
        } 