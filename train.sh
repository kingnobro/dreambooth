TARGET="man"
MODEL_NAME="runwayml/stable-diffusion-v1-5"
INSTANCE_DIR="train_images/$TARGET"
CLASS_DIR="class_images/$TARGET"
OUTPUT_DIR="models/$TARGET"

accelerate launch --config_file config/default_config.yaml train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="sks man character" \
  --class_prompt="man character" \
  --resolution=512 \
  --train_batch_size=6 \
  --gradient_checkpointing \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=400 \
  --checkpointing_steps=500 \
