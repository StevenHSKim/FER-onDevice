import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import gc
from inference_utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Facial Expression Recognition Inference for Jetson Nano')
    
    # Paths
    parser.add_argument('--raf_path', type=str, default='./datasets/raf-basic/', 
                        help='Path to RAF-DB dataset')
    parser.add_argument('--output_dir', type=str, default='./results/',
                        help='Directory to save results')
    
    # Model selection and checkpoints
    parser.add_argument('--model_type', type=str, default='adaDF', 
                    choices=['adaDF', 'dan', 'ddamfn', 'gsdnet', 'lnsunet', 'poster', 'posterv2', 'emotiefflib', 's2d'], 
                    help='Model type to use for inference')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    
    # POSTER 특정 파라미터 추가
    parser.add_argument('--poster_model_type', type=str, default='large',
                    choices=['small', 'base', 'large'],
                    help='POSTER model size (small, base, large)')
    
    # Model specific parameters
    parser.add_argument('--num_classes', type=int, default=7,
                        help='Number of emotion classes')
    parser.add_argument('--drop_rate', type=float, default=0.0,
                        help='Dropout rate for AdaDF model')
    parser.add_argument('--num_head', type=int, default=4,
                        help='Number of attention heads for DAN/DDAMFN models')
    
    # Inference parameters
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference (default=1 for Jetson)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers')
    
    # Input selection
    parser.add_argument('--input_file', type=str, default=None,
                        help='Path to text file with list of image files to process (optional)')
    parser.add_argument('--limit_samples', type=int, default=None,
                        help='Limit number of samples to process (optional, for testing)')
    
    # Optimization options for Jetson
    parser.add_argument('--use_fp16', action='store_true', default=False,
                        help='Use FP16 precision for inference (if supported)')
    parser.add_argument('--measure_time', action='store_true', default=True,
                        help='Measure inference time')
    parser.add_argument('--clear_memory', action='store_true', default=True,
                        help='Clear memory after each batch (slower but helps with OOM)')
    
    return parser.parse_args()

def inference_jetson(model, dataloader, model_type, device, use_fp16=False, measure_time=True, clear_memory=True):
    results = []
    inference_times = []
    
    # Set up for FP16 inference if requested and supported
    if use_fp16 and hasattr(torch.cuda, 'amp') and device.type == 'cuda':
        print("Using FP16 precision for inference")
        amp_enabled = True
    else:
        amp_enabled = False
    
    with torch.no_grad():
        for images, file_names in tqdm(dataloader, desc="Processing images"):
            images = images.to(device)
            
            start_time = time.time()
            
            # FP16 inference
            if amp_enabled:
                with torch.cuda.amp.autocast():
                    outputs = process_model_output(model, images, model_type)
            else:
                outputs = process_model_output(model, images, model_type)
            
            end_time = time.time()
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Store results
            batch_results = [(file_name, pred.item()) for file_name, pred in zip(file_names, preds)]
            results.extend(batch_results)
            
            # Clear memory if requested
            if clear_memory:
                del images, outputs, preds
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
    
    # Report inference statistics
    if measure_time and inference_times:
        avg_time = sum(inference_times) / len(inference_times)
        print(f"Average inference time per batch: {avg_time:.4f} seconds")
        print(f"FPS: {1.0/avg_time:.2f}")
    
    return results

def process_model_output(model, images, model_type):
    """Process model outputs based on model type"""
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
    
    return outputs

def main():
    args = parse_args()
    
    # Detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        # Print CUDA memory info
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Available GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB used")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load input file list if provided
    input_file_list = None
    if args.input_file and os.path.exists(args.input_file):
        with open(args.input_file, 'r') as f:
            input_file_list = [line.strip() for line in f.readlines()]
            if args.limit_samples:
                input_file_list = input_file_list[:args.limit_samples]
        print(f"Loaded {len(input_file_list)} files from {args.input_file}")
    
    # Load the appropriate model and transform
    print(f"Loading {args.model_type} model from {args.checkpoint_path}")
    
    try:
        if args.model_type == 'adaDF':
            model = load_adaDF_model(args.checkpoint_path, args.num_classes, args.drop_rate, device)
            transform = get_adaDF_transform()
            dataset = RafDataset(args.raf_path, input_file_list, transform=transform)
        elif args.model_type == 'dan':
            model = load_dan_model(args.checkpoint_path, args.num_head, device)
            transform = get_dan_transform()
            dataset = RafDataset(args.raf_path, input_file_list, transform=transform)
        elif args.model_type == 'ddamfn':
            model = load_ddamfn_model(args.checkpoint_path, args.num_head, args.num_classes, device)
            transform = get_ddamfn_transform()
            dataset = RafDataset(args.raf_path, input_file_list, transform=transform)
        elif args.model_type == 'gsdnet':
            model = load_gsdnet_model(args.checkpoint_path, args.num_classes, device)
            transform = get_gsdnet_transform()
            dataset = RafDataset(args.raf_path, input_file_list, transform=transform)
        elif args.model_type == 'lnsunet':
            model = load_lnsunet_model(args.checkpoint_path, device)
            transform = get_lnsunet_transform()
            dataset = RafDataset(args.raf_path, input_file_list, transform=transform)
        elif args.model_type == 'poster':
            model = load_poster_model(args.checkpoint_path, args.poster_model_type, args.num_classes, device)
            transform = get_poster_transform()
            dataset = RafDataset(args.raf_path, input_file_list, transform=transform)
        elif args.model_type == 'posterv2':
            model = load_posterv2_model(args.checkpoint_path, args.num_classes, device)
            transform = get_posterv2_transform()
            dataset = RafDataset(args.raf_path, input_file_list, transform=transform)
        elif args.model_type == 'emotiefflib':
            model = load_emotiefflib_model(args.checkpoint_path, args.num_classes, device)
            transform = get_emotiefflib_transform()
            dataset = RafDataset(args.raf_path, input_file_list, transform=transform)
        elif args.model_type == 's2d':
            model = load_s2d_model(args.checkpoint_path, args.num_classes, device)
            transform = get_s2d_transform()
            dataset = s2dDataset(args.raf_path, input_file_list, transform=transform)
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Optional: Limit dataset size for testing
    if args.limit_samples and input_file_list is None:
        indices = list(range(min(args.limit_samples, len(dataset))))
        from torch.utils.data import Subset
        dataset = Subset(dataset, indices)
        print(f"Limited to {len(dataset)} samples for testing")
    
    # Create dataloader with pinned memory for faster data transfer but smaller batch size for Jetson
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize model to eval mode
    model.eval()
    
    # Run optimized inference for Jetson
    print(f"Running inference with {args.model_type} model...")
    results = inference_jetson(
        model, 
        dataloader, 
        args.model_type, 
        device, 
        use_fp16=args.use_fp16,
        measure_time=args.measure_time,
        clear_memory=args.clear_memory
    )
    
    # Convert results to DataFrame
    emotion_map = get_emotion_map()
    results_df = pd.DataFrame(results, columns=['File', 'PredictionId'])
    results_df['Emotion'] = results_df['PredictionId'].map(emotion_map)
    
    # Save results
    output_file = os.path.join(args.output_dir, f"{args.model_type}_predictions_jetson.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Print some statistics
    print("\nEmotion distribution in predictions:")
    emotion_counts = results_df['Emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"{emotion}: {count} ({count/len(results_df)*100:.2f}%)")

if __name__ == "__main__":
    main()