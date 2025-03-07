import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Visualization of FER predictions')
    parser.add_argument('--results_file', type=str, required=True,
                        help='Path to the CSV file with prediction results')
    parser.add_argument('--raf_path', type=str, default='./datasets/raf-basic/',
                        help='Path to RAF-DB dataset')
    parser.add_argument('--output_dir', type=str, default='./visualization/',
                        help='Directory to save visualization results')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of sample images to visualize')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for sample selection')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load prediction results
    results_df = pd.read_csv(args.results_file)
    
    # Set random seed for reproducibility
    np.random.seed(args.random_seed)
    
    # Select random samples for visualization
    sample_indices = np.random.choice(len(results_df), min(args.num_samples, len(results_df)), replace=False)
    sample_results = results_df.iloc[sample_indices]
    
    # Visualization
    emotion_colors = {
        'Neutral': (220, 220, 220),  # Light grey
        'Happy': (70, 200, 70),      # Green
        'Sad': (70, 70, 200),        # Blue
        'Surprise': (200, 200, 70),  # Yellow
        'Fear': (200, 70, 200),      # Purple
        'Disgust': (70, 200, 200),   # Cyan
        'Angry': (200, 70, 70)       # Red
    }
    
    for i, (_, row) in enumerate(sample_results.iterrows()):
        file_name = row['File']
        emotion = row['Emotion']
        
        # Load and process the image
        img_path = os.path.join(args.raf_path, 'Image/aligned', file_name.split(".")[0] + "_aligned.jpg")
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found at {img_path}")
            continue
            
        # Read image using OpenCV
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Add a colored bar at the bottom with the predicted emotion
        color = emotion_colors.get(emotion, (128, 128, 128))
        
        # Create a blank image for the bar
        bar_height = 30
        bar_img = np.ones((bar_height, img.shape[1], 3), dtype=np.uint8) * np.array(color, dtype=np.uint8)
        
        # Add text to the bar
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Prediction: {emotion}"
        textsize = cv2.getTextSize(text, font, 0.7, 2)[0]
        text_x = (bar_img.shape[1] - textsize[0]) // 2
        text_y = (bar_img.shape[0] + textsize[1]) // 2
        cv2.putText(bar_img, text, (text_x, text_y), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Combine image and bar
        combined_img = np.vstack((img, bar_img))
        
        # Save the visualization
        output_path = os.path.join(args.output_dir, f"pred_{i+1}_{file_name.split('.')[0]}.jpg")
        plt.figure(figsize=(8, 8))
        plt.imshow(combined_img)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    
    print(f"Visualizations saved to {args.output_dir}")
    
    # Create a summary visualization of emotion distribution
    plt.figure(figsize=(12, 6))
    
    # Count emotions in the entire result set
    emotion_counts = results_df['Emotion'].value_counts()
    
    # Create a bar chart
    bars = plt.bar(emotion_counts.index, emotion_counts.values, color=[emotion_colors[e] for e in emotion_counts.index])
    
    # Add percentage labels on top of each bar
    for i, bar in enumerate(bars):
        emotion = emotion_counts.index[i]
        count = emotion_counts.values[i]
        percentage = count / len(results_df) * 100
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{percentage:.1f}%', ha='center', va='bottom', rotation=0)
    
    plt.title('Distribution of Predicted Emotions')
    plt.ylabel('Count')
    plt.ylim(0, max(emotion_counts.values) * 1.1)  # Add 10% space above the highest bar
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the distribution plot
    dist_plot_path = os.path.join(args.output_dir, "emotion_distribution.png")
    plt.savefig(dist_plot_path)
    plt.close()
    
    print(f"Emotion distribution plot saved to {dist_plot_path}")

if __name__ == "__main__":
    main()