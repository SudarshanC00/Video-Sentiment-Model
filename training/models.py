import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models
from sklearn.metrics import precision_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

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

def compute_class_weights(dataset):
    emotion_counts = torch.zeros(7)  # Assuming 7 emotion classes
    sentiment_counts = torch.zeros(3)  # Assuming 3 sentiment classes
    skipped = 0
    total = len(dataset)

    print("\Counting class distributions...")
    for i in range(total):
        sample = dataset[i]

        if sample is None:
            skipped += 1
            continue

        emotion_label = sample['emotion_label']
        sentiment_label = sample['sentiment_label']

        emotion_counts[emotion_label] += 1
        sentiment_counts[sentiment_label] += 1

    valid = total - skipped
    print(f"Skipped samples: {skipped}/{total}")

    print("\nClass distribution")
    print("Emotions:")
    emotion_map =  {0:'anger', 1:'disgust', 2:'fear', 3:'joy', 4:'neutral', 5:'sadness', 6:'surprise'}
    for i, count in enumerate(emotion_counts):
        print(f"{emotion_map[i]}: {count/valid:.2f}")

    print("\Sentiments:")
    sentiment_map = {0:'negative', 1:'neutral', 2:'positive'}
    for i, count in enumerate(sentiment_counts):
        print(f"{sentiment_map[i]}: {count/valid:.2f}")

    # Calculate class weights
    emotion_weights = 1.0/ emotion_counts
    sentiment_weights = 1.0/ sentiment_counts

    # Normalise weights
    emotion_weights = emotion_weights/emotion_weights.sum()
    sentiment_weights = sentiment_weights/sentiment_weights.sum()

    return emotion_weights, sentiment_weights


class MultiModalTrainer:
    def __init__(self, model, train_loader, val_loader, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        

        # Log dataset sizes
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        print("\nDataset sizes:")
        print(f"Train dataset size: {train_size:,}")
        print(f"Validation dataset size: {val_size:,}")
        print(f"Batchs per epoch: {len(train_loader):,}")

        timestamp = datetime.now().strftime('%b%d_%H-%M-%S') # e.g., 'Oct01_12-30-45'
        base_dir = '/opt/ml/output/tensorboard' if 'SM_MODEL_DIR' in os.environ else 'runs'
        log_dir = f'{base_dir}/run_{timestamp}'
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0


        #
        self.optimizer = torch.optim.Adam([
            {'params': self.model.text_encoder.parameters(), 'lr': 8e-6},
            {'params': self.model.video_encoder.parameters(), 'lr': 8e-5},
            {'params': self.model.audio_encoder.parameters(), 'lr': 8e-5},
            {'params': self.model.fusion_layer.parameters(), 'lr': 5e-4},
            {'params': self.model.emotion_classifier.parameters(), 'lr': 5e-4},
            {'params': self.model.sentiment_classifier.parameters(), 'lr': 5e-4}
        ], weight_decay= 1e-5)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.1, 
            patience=2
        )
    
        self.current_train_losses = None

        # Calculate class weights
        print("\Calculating class weights...")
        emotion_weights, sentiment_weights = compute_class_weights(train_loader.dataset)

        self.emotion_weights = emotion_weights.to(self.device)
        self.sentiment_weights = sentiment_weights.to(self.device)

        print(f"Emotion weights on device: {self.emotion_weights.device}")
        print(f"Sentiment weights on device: {self.sentiment_weights.device}")


        self.emotion_criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05,
            weight = self.emotion_weights
        )

        self.sentiment_criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05,
            weight = self.sentiment_weights
        )

    def log_metrics(self, losses, metrics=None, phase='train'):
        if phase == "train":
            self.current_train_losses = losses
        else: # Validation Phase
            self.writer.add_scalar(
                'loss/total/train', self.current_train_losses['total'], self.global_step
                )
            self.writer.add_scalar(
                'loss/total/val', losses['total'], self.global_step
            )
            self.writer.add_scalar(
                'loss/emotion/train', self.current_train_losses['emotion'], self.global_step
            )
            self.writer.add_scalar(
                'loss/emotion/val', losses['emotion'], self.global_step
            )
            self.writer.add_scalar(
                'loss/sentiment/train', self.current_train_losses['sentiment'], self.global_step
            )
            self.writer.add_scalar(
                'loss/sentiment/val', losses['sentiment'], self.global_step
            )
        
        if metrics:
            self.writer.add_scalar(
                f'{phase}/emotion_precision', metrics['emotion_precision'], self.global_step
            )
            self.writer.add_scalar(
                f'{phase}/emotion_accuracy', metrics['emotion_accuracy'], self.global_step
            )
            self.writer.add_scalar(
                f'{phase}/sentiment_precision', metrics['sentiment_precision'], self.global_step
            )
            self.writer.add_scalar(
                f'{phase}/sentiment_accuracy', metrics['sentiment_accuracy'], self.global_step
            )
    

    def train_epoch(self):
        self.model.train()
        running_loss = {'total':0, 'emotion':0, 'sentiment':0}

        for batch in self.train_loader:
            # device = next(self.model.parameters()).device
            device = self.device
            text_inputs = {
                'input_ids': batch['text_inputs']['input_ids'].to(device),
                'attention_mask': batch['text_inputs']['attention_mask'].to(device)
            }
            video_frames = batch['video_frames'].to(device)
            audio_features = batch['audio_features'].to(device)
            emotion_labels = batch['emotion_label'].to(device)
            sentiment_labels = batch['sentiment_label'].to(device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(text_inputs, video_frames, audio_features)

            # Compute losses using raw logits
            emotion_loss = self.emotion_criterion(outputs['emotion_output'], emotion_labels)
            sentiment_loss = self.sentiment_criterion(outputs['sentiment_output'], sentiment_labels)
            total_loss = emotion_loss + sentiment_loss

            # Backward pass and calculate gradients
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()

            # Track losses
            running_loss['total'] += total_loss.item()
            running_loss['emotion'] += emotion_loss.item()
            running_loss['sentiment'] += sentiment_loss.item()

            self.log_metrics({
                'total': total_loss.item(),
                'emotion': emotion_loss.item(),
                'sentiment': sentiment_loss.item()
            })

            self.global_step += 1

        return {
            k: v/len(self.train_loader) for k, v in running_loss.items()
        }
    
    def evaluate(self, data_loader, phase='val'):
        self.model.eval()
        losses = {'total':0, 'emotion':0, 'sentiment':0}
        all_emotion_preds = []
        all_sentiment_preds = []
        all_emotion_labels = []
        all_sentiment_labels = []

        with torch.inference_mode():
            for batch in data_loader:
                # device = next(self.model.parameters()).device
                device = self.device
                text_inputs = {
                    'input_ids': batch['text_inputs']['input_ids'].to(device),
                    'attention_mask': batch['text_inputs']['attention_mask'].to(device)
                }
                video_frames = batch['video_frames'].to(device)
                audio_features = batch['audio_features'].to(device)
                emotion_labels = batch['emotion_label'].to(device)
                sentiment_labels = batch['sentiment_label'].to(device)

                # Forward pass
                outputs = self.model(text_inputs, video_frames, audio_features)

                # Compute losses using raw logits
                emotion_loss = self.emotion_criterion(outputs['emotion_output'], emotion_labels)
                sentiment_loss = self.sentiment_criterion(outputs['sentiment_output'], sentiment_labels)
                total_loss = emotion_loss + sentiment_loss

                # Collect predictions and labels
                all_emotion_preds.extend(outputs['emotion_output'].argmax(dim=1).cpu().numpy())
                all_sentiment_preds.extend(outputs['sentiment_output'].argmax(dim=1).cpu().numpy())
                all_emotion_labels.extend(emotion_labels.cpu().numpy())
                all_sentiment_labels.extend(sentiment_labels.cpu().numpy())
    
                # Track losses
                losses['total'] += total_loss.item()
                losses['emotion'] += emotion_loss.item()
                losses['sentiment'] += sentiment_loss.item()

            avg_loss = {k: v/len(data_loader) for k,v in losses.items()}

            # Compute the precision and accuracy
            emotion_precision = precision_score(
                all_emotion_labels, 
                all_emotion_preds, 
                average='weighted'
            )
            emotion_accuracy = accuracy_score(
                all_emotion_labels, 
                all_emotion_preds
            )
            sentiment_precision = precision_score(
                all_sentiment_labels, 
                all_sentiment_preds, 
                average='weighted'
            )
            sentiment_accuracy = accuracy_score(
                all_sentiment_labels, 
                all_sentiment_preds
            )

            self.log_metrics(avg_loss, {
                'emotion_precision': emotion_precision,
                'emotion_accuracy': emotion_accuracy,
                'sentiment_precision': sentiment_precision,
                'sentiment_accuracy': sentiment_accuracy
            }, phase = phase)

            if phase == 'val':
                self.scheduler.step(avg_loss['total'])

            return avg_loss,{
                'emotion_precision': emotion_precision,
                'emotion_accuracy': emotion_accuracy,
                'sentiment_precision': sentiment_precision,
                'sentiment_accuracy': sentiment_accuracy
            }