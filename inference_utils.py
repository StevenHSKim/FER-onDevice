import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Common dataset class that can be used for all models
class RafDataset(Dataset):
    def __init__(self, raf_path, file_list=None, transform=None):
        self.transform = transform
        self.raf_path = raf_path
        
        if file_list is not None:
            # Use provided file list
            self.file_names = file_list
            self.file_paths = [os.path.join(self.raf_path, 'Image/aligned', f.split(".")[0] + "_aligned.jpg") 
                              for f in self.file_names]
        else:
            # Load all files from the directory
            df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), 
                           sep=' ', header=None, names=['name', 'label'])
            self.file_names = df['name'].values
            self.labels = df['label'].values - 1  # 0-based indexing
            self.file_paths = [os.path.join(self.raf_path, 'Image/aligned', f.split(".")[0] + "_aligned.jpg") 
                              for f in self.file_names]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, self.file_names[idx]

# Model loader functions
def load_adaDF_model(checkpoint_path, num_classes=7, drop_rate=0.0, device=None):
    from model import create_model  # This should be imported from the Ada-DF project
    
    model = create_model(num_classes, drop_rate).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    return model

def load_dan_model(checkpoint_path, num_head=4, device=None):
    from networks.dan import DAN  # This should be imported from the DAN project
    
    model = DAN(num_head=num_head)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if model was saved with DataParallel
    if all([k.startswith('module.') for k in checkpoint['model_state_dict'].keys()]):
        # Remove 'module.' prefix
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    return model

def load_ddamfn_model(checkpoint_path, num_head=2, num_class=7, device=None):
    from networks.DDAM import DDAMNet  # This should be imported from the DDAMFN project
    
    model = DDAMNet(num_class=num_class, num_head=num_head)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if model was saved with DataParallel
    if all([k.startswith('module.') for k in checkpoint['model_state_dict'].keys()]):
        # Remove 'module.' prefix
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    return model

# Create transform functions for each model
def get_adaDF_transform():
    # Assuming AdaDF uses standard transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def get_dan_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def get_ddamfn_transform():
    transform = transforms.Compose([
        transforms.Resize((112, 112)),  # DDAMFN uses smaller input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

# GSDNet 모델 로더 함수
def load_gsdnet_model(checkpoint_path, num_classes=7, device=None):
    from FER_Models.GSDNet.model.gsdnet import GSDNet  # GSDNet 모델 임포트
    
    model = GSDNet(num_class=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 필요한 경우 DataParallel 처리
    if all([k.startswith('module.') for k in checkpoint['model_state_dict'].keys()]):
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    return model

# LNSU-Net 모델 로더 함수
def load_lnsunet_model(checkpoint_path, device=None):
    from backbones.swin import SwinTransformer
    from expression.models import SwinTransFER
    
    swin = SwinTransformer(num_classes=512)
    model = SwinTransFER(swin=swin, swin_num_features=768, num_classes=7, cam=True)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # DataParallel 처리
    if all([k.startswith('module.') for k in checkpoint['state_dict_model'].keys()]):
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict_model'].items()}
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint['state_dict_model'])
    
    model.to(device)
    model.eval()
    
    return model

# POSTER 모델 로더 함수
def load_poster_model(checkpoint_path, model_type='large', num_classes=7, device=None):
    from models.emotion_hyp import pyramid_trans_expr  # POSTER 모델 임포트
    
    model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=model_type)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # DataParallel 처리
    if all([k.startswith('module.') for k in checkpoint['model_state_dict'].keys()]):
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    return model

# POSTER++ 모델 로더 함수
def load_posterv2_model(checkpoint_path, num_classes=7, device=None):
    from models.PosterV2_7cls import pyramid_trans_expr2  # POSTER++ 모델 임포트
    
    model = pyramid_trans_expr2(img_size=112, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # DataParallel 처리
    if all([k.startswith('module.') for k in checkpoint['model_state_dict'].keys()]):
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    return model

# emotiefflib 모델 로더 함수
def load_emotiefflib_model(checkpoint_path, num_classes=7, device=None):
    import timm
    
    # 모델 생성
    model = timm.create_model('tf_emotiefflib_b0_ns', pretrained=False)
    model.classifier = torch.nn.Sequential(torch.nn.Linear(in_features=1280, out_features=num_classes))
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 전체 모델이 저장된 경우(위 Jupyter에서 `torch.save(model, PATH)` 방식으로 저장됨)
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    # state_dict만 저장된 경우
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    # 단순 state_dict만 저장된 경우
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model

# s2d 모델 로더 함수
def load_s2d_model(checkpoint_path, num_classes=7, device=None):
    import torch
    from timm.models import create_model
    
    # 모델 생성
    model = create_model(
        'vit_base_patch16_224',
        pretrained=False,
        num_classes=num_classes,
        adapter_scale=0.25,  # 기본 값 사용
        head_dropout_ratio=0.5,  # 기본 값 사용
        num_frames=16,  # 대표적인 설정 사용
        in_chans_l=128  # 기본 값 사용
    )
    
    # Landmark 모델은 사용하지 않고 메인 모델만 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 모델 형태에 따라 다르게 로드
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model

# 각 모델에 대한 transform 함수들 추가
def get_gsdnet_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def get_lnsunet_transform():
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform

def get_poster_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def get_posterv2_transform():
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def get_emotiefflib_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def get_s2d_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

# s2d는 비디오 처리를 위한 특별한 데이터셋 클래스가 필요합니다
# 하지만 단일 이미지 추론을 위해 간소화된 버전 구현
class s2dDataset(torch.utils.data.Dataset):
    def __init__(self, raf_path, file_list=None, transform=None):
        self.transform = transform
        self.raf_path = raf_path
        
        if file_list is not None:
            # Use provided file list
            self.file_names = file_list
            self.file_paths = [os.path.join(self.raf_path, 'Image/aligned', f.split(".")[0] + "_aligned.jpg") 
                              for f in self.file_names]
        else:
            # Load all files from the directory
            df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), 
                           sep=' ', header=None, names=['name', 'label'])
            self.file_names = df['name'].values
            self.labels = df['label'].values - 1  # 0-based indexing
            self.file_paths = [os.path.join(self.raf_path, 'Image/aligned', f.split(".")[0] + "_aligned.jpg") 
                              for f in self.file_names]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
            
            # s2d는 비디오 입력을 기대하므로 단일 이미지를 16프레임으로 복제
            # [C, H, W] -> [C, T, H, W] -> [T, C, H, W]
            image = image.unsqueeze(1).repeat(1, 16, 1, 1).permute(1, 0, 2, 3)
            
        return image, self.file_names[idx]

# Common inference function
# 기존 inference 함수를 수정하여 추가 모델을 지원하도록 함
def inference(model, dataloader, model_type, device):
    results = []
    
    with torch.no_grad():
        for images, file_names in dataloader:
            images = images.to(device)
            
            if model_type == 'adaDF':
                # AdaDF model returns (outputs_1, outputs_2, attention_weights)
                outputs, _, _ = model(images)
                
            elif model_type == 'dan':
                # DAN model returns (out, feat, heads)
                outputs, _, _ = model(images)
                
            elif model_type == 'ddamfn':
                # DDAMFN model returns (out, feat, heads)
                outputs, _, _ = model(images)
                
            elif model_type == 'gsdnet':
                # GSDNet model returns (pred, feat, pred_teacher, feat_teacher)
                outputs, _, _, _ = model(images)
                # 마지막 classifier의 출력(가장 정확도 높은 것)만 사용
                outputs = outputs[3]
                
            elif model_type == 'lnsunet':
                # LNSU-Net model returns (output, heatmap)
                outputs, _ = model(images)
                
            elif model_type == 'poster':
                # POSTER model returns (outputs, features)
                outputs, _ = model(images)
                
            elif model_type == 'posterv2':
                # POSTER++ model returns 표현식 출력값
                outputs = model(images)
                
            elif model_type == 'emotiefflib':
                # emotiefflib 모델은 일반적인 분류 출력만 반환
                outputs = model(images)
                
            elif model_type == 's2d':
                # s2d 모델의 출력 처리
                outputs = model(images)
                if isinstance(outputs, tuple):
                    # 일부 모델은 여러 출력을 반환할 수 있음
                    outputs = outputs[0]
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Store results
            batch_results = [(file_name, pred.item()) for file_name, pred in zip(file_names, preds)]
            results.extend(batch_results)
    
    return results

# Emotion mapping
def get_emotion_map():
    return {
        0: 'Neutral',
        1: 'Happy',
        2: 'Sad',
        3: 'Surprise',
        4: 'Fear',
        5: 'Disgust',
        6: 'Angry'
    }

# Ensemble predictions from multiple models
def ensemble_predictions(predictions_list, method='voting'):
    """
    Combine predictions from multiple models
    
    Args:
        predictions_list: List of prediction dictionaries, each from a different model
        method: 'voting' for majority vote, 'average' for average of probabilities
        
    Returns:
        Dictionary of filename to prediction after ensemble
    """
    if len(predictions_list) == 1:
        return predictions_list[0]
    
    # Create a dictionary to store all predictions for each file
    ensemble_dict = {}
    
    # Collect all predictions for each file
    for predictions in predictions_list:
        for file_name, pred in predictions:
            if file_name not in ensemble_dict:
                ensemble_dict[file_name] = []
            ensemble_dict[file_name].append(pred)
    
    # Apply voting method
    if method == 'voting':
        final_predictions = []
        for file_name, preds in ensemble_dict.items():
            # Get most common prediction
            unique_values, counts = np.unique(preds, return_counts=True)
            final_pred = unique_values[np.argmax(counts)]
            final_predictions.append((file_name, final_pred))
    
    return final_predictions