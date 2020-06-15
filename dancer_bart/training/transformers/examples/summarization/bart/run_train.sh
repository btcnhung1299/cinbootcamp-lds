export CURRENT_DIR=${PWD}
export OUTPUT_DIR="./_hybrid_ckpt"
export HIP_VISIBLE_DEVICES=1

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../../":"${PYTHONPATH}"

python3 finetune.py \
--data_dir=../../../../encoding/ \
--model_name_or_path=bart-large \
--max_source_length=1024 \
--max_target_length=256 \
--learning_rate=3e-7 \
--train_batch_size=2 \
--output_dir="$OUTPUT_DIR" \
--resume_from_checkpoint=../../../../ckpt/hybrid/hybrid_epoch1.ckpt \
--do_train  $@
