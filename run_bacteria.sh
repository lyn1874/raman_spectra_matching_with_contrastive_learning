#!/bin/bash
trap "exit" INT
raman_type=${1:-bacteria_random_reference_finetune}
repeat_g=${2:-0}
dir2save_ckpt=${3:-exp_data}  # USER-DEFINE
dir2load_data=${4:-data_group}  # USER-DEFINE
# ver=${5:-0}
ver=0
random_leave_one_out=true

lr_init=0.00005
lr_schedule=cosine
stem_kernel=21
stem_max_dim=64
depth=128
dist_aggregate=wave_channel_dot_L1

dropout=true
within_dropout=true
l2_regu_para=0.001

balanced=true
separable_act=true
balanced_alpha=0.5 #0.1

siamese_version=0
batch_size=92
max_epoch=1

check_distance_on_datasplit=false
check_ratio_on_datasplit=false

for repeat in $repeat_g
do
  echo "Experiment with pure random leave one out" + "$repeat"
  python3 train_inception.py --version "$ver" --dataset BACTERIA --raman_type "$raman_type" \
                    --batch_size "$batch_size" --max_epoch "$max_epoch" \
                    --learning_rate_init "$lr_init" \
                    --distance_aggregation "$dist_aggregate" \
                    --stem_kernel "$stem_kernel" \
                    --lr_schedule "$lr_schedule" --l2_regu_para "$l2_regu_para" \
                    --depth "$depth" --dropout "$dropout" \
                    --neg_shift_scale 5 --stem_max_dim "$stem_max_dim" --within_dropout "$within_dropout" \
                    --balanced "$balanced" --alpha "$balanced_alpha" \
                    --siamese_version "$siamese_version" \
                    --random_leave_one_out "$random_leave_one_out" --separable_act "$separable_act" \
                    --check_distance_on_datasplit "$check_distance_on_datasplit" --repeat_on_python false \
                    --repeat_time "$repeat" --check_ratio_on_datasplit "$check_ratio_on_datasplit" \
                    --dir2load_data "$dir2load_data" --dir2save_ckpt "$dir2save_ckpt"
done

