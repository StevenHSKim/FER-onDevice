import os
import sys
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from inference_utils import *

# Make sure the inference_utils.py is in the same directory or in the Python path
from inference_utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Facial Expression Recognition Inference')
    
    # Paths
    parser.add_argument('--raf_path', type=str, default='./datasets/raf-basic/', 
                        help='Path to RAF-DB dataset')
    parser.add_argument('--output_dir', type=str, default='./results/',
                        help='Directory to save results')
    
    # Model selection and checkpoints
    parser.add_argument('--model_type', type=str, default='adaDF', 
                    choices=['adaDF', 'dan', 'ddamfn', 'gsdnet', 'lnsunet', 'poster', 'posterv2', 'emotiefflib', 's2d', 'ensemble'], 
                    help='Model type to use for inference')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint(s). For ensemble, provide comma-separated paths')
    parser.add_argument('--model_types', type=str, default=None,
                        help='For ensemble, comma-separated list of model types corresponding to checkpoints')
    
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
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Specify GPU device(s) to use')
    
    # Input selection
    parser.add_argument('--input_file', type=str, default=None,
                        help='Path to text file with list of image files to process (optional)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load input file list if provided
    input_file_list = None
    if args.input_file and os.path.exists(args.input_file):
        with open(args.input_file, 'r') as f:
            input_file_list = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(input_file_list)} files from {args.input_file}")
    
    # Ensemble mode
    if args.model_type == 'ensemble':
        if not args.model_types:
            raise ValueError("For ensemble mode, please provide --model_types as comma-separated list")
        
        checkpoint_paths = args.checkpoint_path.split(',')
        model_types = args.model_types.split(',')
        
        if len(checkpoint_paths) != len(model_types):
            raise ValueError("Number of checkpoints must match number of model types")
        
        all_predictions = []
        
        for i, (checkpoint_path, model_type) in enumerate(zip(checkpoint_paths, model_types)):
            print(f"Loading model {i+1}/{len(checkpoint_paths)}: {model_type} from {checkpoint_path}")
            
            # Load the appropriate model and transform
            if model_type == 'adaDF':
                model = load_adaDF_model(checkpoint_path, args.num_classes, args.drop_rate, device)
                transform = get_adaDF_transform()
                dataset = RafDataset(args.raf_path, input_file_list, transform=transform)
            elif model_type == 'dan':
                model = load_dan_model(checkpoint_path, args.num_head, device)
                transform = get_dan_transform()
                dataset = RafDataset(args.raf_path, input_file_list, transform=transform)
            elif model_type == 'ddamfn':
                model = load_ddamfn_model(checkpoint_path, args.num_head, args.num_classes, device)
                transform = get_ddamfn_transform()
                dataset = RafDataset(args.raf_path, input_file_list, transform=transform)
            elif model_type == 'gsdnet':
                model = load_gsdnet_model(checkpoint_path, args.num_classes, device)
                transform = get_gsdnet_transform()
                dataset = RafDataset(args.raf_path, input_file_list, transform=transform)
            elif model_type == 'lnsunet':
                model = load_lnsunet_model(checkpoint_path, device)
                transform = get_lnsunet_transform()
                dataset = RafDataset(args.raf_path, input_file_list, transform=transform)
            elif model_type == 'poster':
                model = load_poster_model(checkpoint_path, args.poster_model_type, args.num_classes, device)
                transform = get_poster_transform()
                dataset = RafDataset(args.raf_path, input_file_list, transform=transform)
            elif model_type == 'posterv2':
                model = load_posterv2_model(checkpoint_path, args.num_classes, device)
                transform = get_posterv2_transform()
                dataset = RafDataset(args.raf_path, input_file_list, transform=transform)
            elif model_type == 'emotiefflib':
                model = load_emotiefflib_model(checkpoint_path, args.num_classes, device)
                transform = get_emotiefflib_transform()
                dataset = RafDataset(args.raf_path, input_file_list, transform=transform)
            elif model_type == 's2d':
                model = load_s2d_model(checkpoint_path, args.num_classes, device)
                transform = get_s2d_transform()
                dataset = s2dDataset(args.raf_path, input_file_list, transform=transform)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Create dataloader
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
            
            # Run inference
            print(f"Running inference with {model_type} model...")
            predictions = inference(model, dataloader, model_type, device)
            all_predictions.append(predictions)
        
        # Combine predictions
        print("Ensembling predictions...")
        results = ensemble_predictions(all_predictions, method='voting')
        
    else:  # Single model mode
        # Load the appropriate model and transform
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
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # Run inference
        print(f"Running inference with {args.model_type} model...")
        results = inference(model, dataloader, args.model_type, device)
    
    # Convert results to DataFrame
    emotion_map = get_emotion_map()
    results_df = pd.DataFrame(results, columns=['File', 'PredictionId'])
    results_df['Emotion'] = results_df['PredictionId'].map(emotion_map)
    
    # Save results
    output_file = os.path.join(args.output_dir, f"{args.model_type}_predictions.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Print some statistics
    print("\nEmotion distribution in predictions:")
    emotion_counts = results_df['Emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"{emotion}: {count} ({count/len(results_df)*100:.2f}%)")

if __name__ == "__main__":
    main()