# Examples of how to run the inference script

# 1. Single model inference (AdaDF)
python run_inference.py \
  --model_type adaDF \
  --checkpoint_path ./checkpoints/adaDF_best.pth \
  --raf_path ./datasets/raf-basic \
  --output_dir ./results \
  --batch_size 32 \
  --gpu 0

# 2. Single model inference (DAN)
python run_inference.py \
  --model_type dan \
  --checkpoint_path ./checkpoints/dan_best.pth \
  --raf_path ./datasets/raf-basic \
  --num_head 4 \
  --output_dir ./results \
  --batch_size 32 \
  --gpu 0

# 3. Single model inference (DDAMFN)
python run_inference.py \
  --model_type ddamfn \
  --checkpoint_path ./checkpoints/ddamfn_best.pth \
  --raf_path ./datasets/raf-basic \
  --num_head 2 \
  --output_dir ./results \
  --batch_size 32 \
  --gpu 0

# 4. Ensemble inference with multiple models
python run_inference.py \
  --model_type ensemble \
  --checkpoint_path ./checkpoints/adaDF_best.pth,./checkpoints/dan_best.pth,./checkpoints/ddamfn_best.pth \
  --model_types adaDF,dan,ddamfn \
  --raf_path ./datasets/raf-basic \
  --output_dir ./results \
  --batch_size 32 \
  --gpu 0

# 5. Using a specific input file list 
python run_inference.py \
  --model_type adaDF \
  --checkpoint_path ./checkpoints/adaDF_best.pth \
  --raf_path ./datasets/raf-basic \
  --input_file ./file_list.txt \
  --output_dir ./results \
  --batch_size 32 \
  --gpu 0

# GSDNet 모델 추론
python run_inference.py \
  --model_type gsdnet \
  --checkpoint_path ./checkpoints/gsdnet_best.pth \
  --raf_path ./datasets/raf-basic \
  --output_dir ./results \
  --batch_size 32 \
  --gpu 0

# LNSU-Net 모델 추론
python run_inference.py \
  --model_type lnsunet \
  --checkpoint_path ./checkpoints/lnsunet_best.pth \
  --raf_path ./datasets/raf-basic \
  --output_dir ./results \
  --batch_size 32 \
  --gpu 0

# POSTER 모델 추론
python run_inference.py \
  --model_type poster \
  --poster_model_type large \
  --checkpoint_path ./checkpoints/poster_best.pth \
  --raf_path ./datasets/raf-basic \
  --output_dir ./results \
  --batch_size 32 \
  --gpu 0

# POSTER++ 모델 추론
python run_inference.py \
  --model_type posterv2 \
  --checkpoint_path ./checkpoints/posterv2_best.pth \
  --raf_path ./datasets/raf-basic \
  --output_dir ./results \
  --batch_size 32 \
  --gpu 0

# EmotiEffLib 모델 추론
python run_inference.py \
  --model_type EmotiEffLib \
  --checkpoint_path ./checkpoints/rafdb_march2021_EmotiEffLib_b0.pt \
  --raf_path ./datasets/raf-basic \
  --output_dir ./results \
  --batch_size 32 \
  --gpu 0

# S2D 모델 추론
python run_inference.py \
  --model_type S2D \
  --checkpoint_path ./checkpoints/S2D_best.pth \
  --raf_path ./datasets/raf-basic \
  --output_dir ./results \
  --batch_size 32 \
  --gpu 0