#!/bin/bash
# -------------------------------
# LoRA-PTI Training Script
# -------------------------------

# Base model to finetune
export MODEL_NAME="runwayml/stable-diffusion-v1-5"

# Directory containing your training dataset
export INSTANCE_DIR="./data/flowers"

# Directory to save checkpoints, logs, and final LoRA weights
export OUTPUT_DIR="./exps/output_dsn"

# Run the LoRA-PTI training script
lora_pti \
  --pretrained_model_name_or_path=$MODEL_NAME  \   # Pretrained Stable Diffusion model
  --instance_data_dir=$INSTANCE_DIR \              # Dataset folder
  --output_dir=$OUTPUT_DIR \                       # Output folder
  --train_text_encoder \                           # Also fine-tune the text encoder
  --resolution=512 \                               # Image resolution for training
  --train_batch_size=1 \                           # Per-GPU batch size
  --gradient_accumulation_steps=4 \               # Effective batch size = batch_size * accumulation
  --scale_lr \                                     # Scale learning rate according to batch size
  --learning_rate_unet=1e-4 \                      # Learning rate for UNet
  --learning_rate_text=1e-5 \                      # Learning rate for text encoder
  --learning_rate_ti=5e-4 \                        # Learning rate for textual inversion embeddings
  --color_jitter \                                 # Random brightness/contrast/color jitter
  --lr_scheduler="linear" \                        # Linear decay of learning rate
  --lr_warmup_steps=0 \                            # No learning rate warmup
  --placeholder_tokens="<s1>|<s2>" \               # Tokens to train in textual inversion
  --use_template="style" \                         # Use template for prompts
  --save_steps=100 \                               # Save model checkpoint every 100 steps
  --max_train_steps_ti=1000 \                      # Maximum training steps for textual inversion
  --max_train_steps_tuning=1000 \                  # Maximum training steps for LoRA tuning
  --perform_inversion=True \                       # Perform textual inversion first
  --clip_ti_decay \                                # Decay TI embeddings over time
  --weight_decay_ti=0.000 \                        # Weight decay for TI embeddings
  --weight_decay_lora=0.001 \                      # Weight decay for LoRA layers
  --continue_inversion \                           # Resume inversion if previously interrupted
  --continue_inversion_lr=1e-4 \
  --device="cuda:0" \
  --lora_rank=1 \
  #  --use_face_segmentation_condition\