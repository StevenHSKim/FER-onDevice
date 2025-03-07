#!/bin/bash
# Jetson Nano 추론 실행 예제

## GSDNet, LNSU-Net, POSTER(small, base, large 모델 모두), POSTER++, S2D 모델 실행 코드는 우선 배제
## 안 돌아갈 것이라 판단됨

# AdaDF 모델 추론 (배치 크기 1로 설정, 메모리 관리)
python run_inference_jetson.py \
  --model_type adaDF \
  --checkpoint_path ./checkpoints/adaDF_best.pth \
  --raf_path ./datasets/raf-basic \
  --output_dir ./results \
  --batch_size 1 \
  --num_workers 2 \
  --clear_memory

# DAN 모델 추론 (Jetson Nano에 최적화)
python run_inference_jetson.py \
  --model_type dan \
  --checkpoint_path ./checkpoints/dan_best.pth \
  --raf_path ./datasets/raf-basic \
  --num_head 4 \
  --output_dir ./results \
  --batch_size 1 \
  --measure_time

# DDAMFN 모델 추론 (특정 이미지 파일만 처리)
python run_inference_jetson.py \
  --model_type ddamfn \
  --checkpoint_path ./checkpoints/ddamfn_best.pth \
  --raf_path ./datasets/raf-basic \
  --num_head 2 \
  --output_dir ./results \
  --input_file ./file_list.txt

# emotiefflib 모델 추론 (샘플 수 제한, 테스트용)
python run_inference_jetson.py \
  --model_type emotiefflib \
  --checkpoint_path ./checkpoints/rafdb_march2021_emotiefflib_b0.pt \
  --raf_path ./datasets/raf-basic \
  --output_dir ./results \
  --limit_samples 50

# 테스트 목적으로 FP16 정밀도를 사용하여 추론 속도 향상 (지원되는 경우)
python run_inference_jetson.py \
  --model_type emotiefflib \
  --checkpoint_path ./checkpoints/rafdb_march2021_emotiefflib_b0.pt \
  --raf_path ./datasets/raf-basic \
  --output_dir ./results \
  --batch_size 1 \
  --use_fp16