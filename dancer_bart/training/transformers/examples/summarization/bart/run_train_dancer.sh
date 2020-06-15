export CURRENT_DIR=${PWD}
export OUTPUT_DIR="./dancer_ckpt"
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
--learning_rate=1e-8 \
--train_batch_size=2 \
--resume_from_checkpoint='../../../../ckpt/dancer/dancer_epoch1.ckpt' \
--output_dir="$OUTPUT_DIR" \
--do_train  $@
