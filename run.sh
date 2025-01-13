#!/usr/bin/env bash
export PYTHONPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

log=loggingtb
base_model=mlp

data=clr_mnist
# bias 1
OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python src/main.py --dataset $data --model $base_model  \
                --optim Adam --test_interval 1 --lr 1e-3 --epochs_stage1 100 --epochs_stage2 900 --norm_input 0\
                --beta_1 0.15 --beta_2 0.15 --beta_1_test 0.15 --beta_2_test 0.15 --hidden_dim 64\
                --train_batch_size 256 --valid_batch_size 256 --test_batch_size 256\
                --eps_type fair_acc --tracker_bz 10 --eps_update_interval 30 --train_eps_epochs 600 --epslr 0.0001 --reinit 1 --last_several_layers 0 \
                --if_version torch --approx_type gumbel  --weight_decay 0.0005 &
sleep 5
# bias 2
OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python src/main.py --dataset $data --model $base_model  \
                --optim Adam --test_interval 1 --lr 1e-3 --epochs_stage1 100 --epochs_stage2 900 --norm_input 0\
                --beta_1 0.85 --beta_2 0.15 --beta_1_test 0.85 --beta_2_test 0.15 --hidden_dim 64\
                --train_batch_size 256 --valid_batch_size 256 --test_batch_size 256\
                --eps_type fair_acc --tracker_bz 10 --eps_update_interval 30 --train_eps_epochs 600 --epslr 0.0001 --reinit 1 --last_several_layers 0 \
                --if_version torch --approx_type gumbel --weight_decay 0.0005 &

sleep 5


# bias 3
OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python src/main.py --dataset $data --model $base_model  \
                --optim Adam --test_interval 1 --lr 1e-3 --epochs_stage1 100 --epochs_stage2 900 --norm_input 0\
                --beta_1 0.5 --beta_2 0.5 --beta_1_test 0.5 --beta_2_test 0.5  \
                --class_beta_1 0.2 --class_beta_2 0.8 --class_beta_1_test 0.2  --class_beta_2_test 0.8 \
                --hidden_dim 64\
                --train_batch_size 256 --valid_batch_size 256 --test_batch_size 256\
                --eps_type fair_acc --tracker_bz 10 --eps_update_interval 30 --train_eps_epochs 600 --epslr 0.0001 --reinit 1 --last_several_layers 0 \
                --if_version torch --approx_type gumbel  --weight_decay 0.0005 &

sleep 5

# bias 1
OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python src/main.py --dataset $data --model $base_model  \
                --optim Adam --test_interval 1 --lr 1e-4 --epochs_stage1 100 --epochs_stage2 900 --norm_input 0\
                --beta_1 0.15 --beta_2 0.15 --beta_1_test 0.15 --beta_2_test 0.15 --hidden_dim 64\
                --train_batch_size 256 --valid_batch_size 256 --test_batch_size 256\
                --eps_type fair_acc --tracker_bz 10 --eps_update_interval 30 --train_eps_epochs 600 --epslr 0.0001 --reinit 1 --last_several_layers 0 \
                --if_version torch --approx_type gumbel  --weight_decay 0.0005 &
sleep 5
# bias 2
OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python src/main.py --dataset $data --model $base_model  \
                --optim Adam --test_interval 1 --lr 1e-4 --epochs_stage1 100 --epochs_stage2 900 --norm_input 0\
                --beta_1 0.85 --beta_2 0.15 --beta_1_test 0.85 --beta_2_test 0.15 --hidden_dim 64\
                --train_batch_size 256 --valid_batch_size 256 --test_batch_size 256\
                --eps_type fair_acc --tracker_bz 10 --eps_update_interval 30 --train_eps_epochs 600 --epslr 0.0001 --reinit 1 --last_several_layers 0 \
                --if_version torch --approx_type gumbel  --weight_decay 0.0005 &

sleep 5


# bias 3
OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python src/main.py --dataset $data --model $base_model  \
                --optim Adam --test_interval 1 --lr 1e-4 --epochs_stage1 100 --epochs_stage2 900 --norm_input 0\
                --beta_1 0.5 --beta_2 0.5 --beta_1_test 0.5 --beta_2_test 0.5  \
                --class_beta_1 0.2 --class_beta_2 0.8 --class_beta_1_test 0.2  --class_beta_2_test 0.8 \
                --hidden_dim 64\
                --train_batch_size 256 --valid_batch_size 256 --test_batch_size 256\
                --eps_type fair_acc --tracker_bz 10 --eps_update_interval 30 --train_eps_epochs 600 --epslr 0.0001 --reinit 1 --last_several_layers 0 \
                --if_version torch --approx_type gumbel  --weight_decay 0.0005 &

sleep 5



## bias 1
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python src/cfair.py --dataset $data --model $base_model  \
#                --optim SGD --test_interval 1 --lr 1e-3 --eps_type none --epoch 1000 --norm_input 0 --tracker_bz 400\
#                --beta_1 0.15 --beta_2 0.15 --beta_1_test 0.15 --beta_2_test 0.15 --hidden_dim 64\
#                --train_batch_size 256 --valid_batch_size 256 --test_batch_size 256&
#sleep 5
## bias 2
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python src/cfair.py --dataset $data --model $base_model  \
#                --optim SGD --test_interval 1 --lr 1e-3 --eps_type none --epoch 1000 --norm_input 0 --tracker_bz 400\
#                --beta_1 0.85 --beta_2 0.15 --beta_1_test 0.85 --beta_2_test 0.15 --hidden_dim 64\
#                --train_batch_size 256 --valid_batch_size 256 --test_batch_size 256&
#
#sleep 5
#
#
## bias 3
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python src/cfair.py --dataset $data --model $base_model  \
#                --optim SGD --test_interval 1 --lr 1e-3 --eps_type none --epoch 1000 --norm_input 0 --tracker_bz 400\
#                --beta_1 0.5 --beta_2 0.5 --beta_1_test 0.5 --beta_2_test 0.5  \
#                --class_beta_1 0.2 --class_beta_2 0.8 --class_beta_1_test 0.2  --class_beta_2_test 0.8 \
#                --hidden_dim 64\
#                --train_batch_size 256 --valid_batch_size 256 --test_batch_size 256&
#
#sleep 5
#
#base_model=lenet
## bias 1
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python src/cfair.py --dataset $data --model $base_model  \
#                --optim SGD --test_interval 1 --lr 1e-3 --eps_type none --epoch 1000 --norm_input 0 --tracker_bz 400\
#                --beta_1 0.15 --beta_2 0.15 --beta_1_test 0.15 --beta_2_test 0.15 --hidden_dim 64\
#                --train_batch_size 256 --valid_batch_size 256 --test_batch_size 256&
#sleep 5
# bias 2
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python src/domind.py --dataset $data --model $base_model  \
#                --optim SGD --test_interval 1 --lr 1e-3 --eps_type none --epoch 1000 --norm_input 0 --tracker_bz 400\
#                --beta_1 0.85 --beta_2 0.15 --beta_1_test 0.85 --beta_2_test 0.15 --hidden_dim 64\
#                --train_batch_size 256 --valid_batch_size 256 --test_batch_size 256&
#
#sleep 5
#
#
## bias 3
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python src/domind.py --dataset $data --model $base_model  \
#                --optim SGD --test_interval 1 --lr 1e-3 --eps_type none --epoch 1000 --norm_input 0 --tracker_bz 400\
#                --beta_1 0.5 --beta_2 0.5 --beta_1_test 0.5 --beta_2_test 0.5  \
#                --class_beta_1 0.2 --class_beta_2 0.8 --class_beta_1_test 0.2  --class_beta_2_test 0.8 \
#                --hidden_dim 64\
#                --train_batch_size 256 --valid_batch_size 256 --test_batch_size 256&

#sleep 5

## bias 1
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python src/basemodel.py --dataset $data --model $base_model  \
#                --optim SGD --test_interval 1 --lr 1e-3 --eps_type none --epoch 1000 --norm_input 0 --tracker_bz 400\
#                --beta_1 0.15 --beta_2 0.15 --beta_1_test 0.15 --beta_2_test 0.15 --hidden_dim 64\
#                --train_batch_size 256 --valid_batch_size 256 --test_batch_size 256&
#sleep 5
## bias 2
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python src/basemodel.py --dataset $data --model $base_model  \
#                --optim SGD --test_interval 1 --lr 1e-3 --eps_type none --epoch 1000 --norm_input 0 --tracker_bz 400\
#                --beta_1 0.85 --beta_2 0.15 --beta_1_test 0.85 --beta_2_test 0.15 --hidden_dim 64\
#                --train_batch_size 256 --valid_batch_size 256 --test_batch_size 256&
#
#sleep 5
#
#
## bias 3
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python src/basemodel.py --dataset $data --model $base_model  \
#                --optim SGD --test_interval 1 --lr 1e-3 --eps_type none --epoch 1000 --norm_input 0 --tracker_bz 400\
#                --beta_1 0.5 --beta_2 0.5 --beta_1_test 0.5 --beta_2_test 0.5  \
#                --class_beta_1 0.2 --class_beta_2 0.8 --class_beta_1_test 0.2  --class_beta_2_test 0.8 \
#                --hidden_dim 64\
#                --train_batch_size 256 --valid_batch_size 256 --test_batch_size 256&


#sleep 5
#

#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python src/basemodel.py --dataset $data --model $base_model  \
#                --optim SGD --test_interval 1 --lr 1e-2 --eps_type none --epoch 500 --norm_input 0\
#                --train_batch_size 256 --valid_batch_size 256 --test_batch_size 256&
#sleep 5
##
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python src/basemodel.py --dataset $data --model $base_model  \
#                --optim SGD --test_interval 1 --lr 1e-3 --eps_type none --epoch 500 --norm_input 0\
#                --train_batch_size 256 --valid_batch_size 256 --test_batch_size 256&
#sleep 5
#
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python src/basemodel.py --dataset $data --model $base_model  \
#                --optim SGD --test_interval 1 --lr 1e-2 --eps_type none --epoch 500 --norm_input 0\
#                --train_batch_size 128 --valid_batch_size 128 --test_batch_size 128&
#sleep 5
##
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python src/basemodel.py --dataset $data --model $base_model  \
#                --optim SGD --test_interval 1 --lr 1e-3 --eps_type none --epoch 500 --norm_input 0\
#                --train_batch_size 128 --valid_batch_size 128 --test_batch_size 128&
#sleep 5
#
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python src/basemodel.py --dataset $data --model $base_model  \
#                --optim SGD --test_interval 1 --lr 1e-2 --eps_type none --epoch 500 --norm_input 0\
#                --train_batch_size 1024 --valid_batch_size 1024 --test_batch_size 1024&
#sleep 5
##
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python src/basemodel.py --dataset $data --model $base_model  \
#                --optim SGD --test_interval 1 --lr 1e-3 --eps_type none --epoch 500 --norm_input 0\
#                --train_batch_size 1024 --valid_batch_size 1024 --test_batch_size 1024&
#sleep 5
#
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python src/basemodel.py --dataset $data --model $base_model  \
#                --optim SGD --test_interval 1 --lr 1e-2 --eps_type none --epoch 500 --norm_input 1\
#                --train_batch_size 128 --valid_batch_size 128 --test_batch_size 128&
#sleep 5
#
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python src/basemodel.py --dataset $data --model $base_model  \
#                --optim SGD --test_interval 1 --lr 1e-3 --eps_type none --epoch 500 --norm_input 1\
#                --train_batch_size 128 --valid_batch_size 128 --test_batch_size 128&
#sleep 5
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python src/basemodel.py --dataset $data --model $base_model  \
#                --optim SGD --test_interval 1 --lr 1e-2 --eps_type none --epoch 500 --norm_input 1\
#                --train_batch_size 1024 --valid_batch_size 1024 --test_batch_size 1024&
#sleep 5
##
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python src/basemodel.py --dataset $data --model $base_model  \
#                --optim SGD --test_interval 1 --lr 1e-3 --eps_type none --epoch 500 --norm_input 1\
#                --train_batch_size 1024 --valid_batch_size 1024 --test_batch_size 1024&
#sleep 5
#
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python src/basemodel.py --dataset $data --model $base_model  \
#                --optim SGD --test_interval 1 --lr 1e-2 --eps_type none --epoch 500 --norm_input 1\
#                --train_batch_size 256 --valid_batch_size 256 --test_batch_size 256&
#sleep 5
##
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python src/basemodel.py --dataset $data --model $base_model  \
#                --optim SGD --test_interval 1 --lr 1e-3 --eps_type none --epoch 500 --norm_input 1\
#                --train_batch_size 256 --valid_batch_size 256 --test_batch_size 256&
#sleep 5

#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python src/domind.py --dataset $data --model $base_model  \
#                --optim SGD --test_interval 1 --lr 1e-2 --eps_type none --epoch 200 --norm_input 0\
#                --train_batch_size 256 --valid_batch_size 256&
#sleep 5
#
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python src/domind.py --dataset $data --model $base_model  \
#                --optim SGD --test_interval 1 --lr 1e-3 --eps_type none --epoch 200 --norm_input 0\
#                --train_batch_size 256 --valid_batch_size 256&
#sleep 5
#
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python src/domind.py --dataset $data --model $base_model  \
#                --optim SGD --test_interval 1 --lr 1e-2 --eps_type none --epoch 200 --norm_input 1\
#                --train_batch_size 256 --valid_batch_size 256&
#sleep 5
#
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python src/domind.py --dataset $data --model $base_model  \
#                --optim SGD --test_interval 1 --lr 1e-3 --eps_type none --epoch 200 --norm_input 1\
#                --train_batch_size 256 --valid_batch_size 256&



#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python src/cfair.py --dataset $data --model $base_model  \
#                --epoch 200 --test_interval 1 --lr 0.0002 --beta_1 0.15 --beta_2 0.15 --beta_1_test 0.15 --beta_2_test 0.15  \
#                --hidden_dim 64  --train_batch_size 256 --valid_batch_size 256 &
#sleep 5

#
## bias 2
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python src/cfair.py --dataset $data --model $base_model  \
#                --epoch 200 --test_interval 1 --lr 0.0002 --beta_1 0.85 --beta_2 0.15 --beta_1_test 0.85 --beta_2_test 0.15  \
#                --hidden_dim 64  --train_batch_size 256 --valid_batch_size 256 &
#
#sleep 5
#
#
## bias 3
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python src/cfair.py --dataset $data --model $base_model  \
#                --epoch 200 --test_interval 1 --lr 0.0002 --beta_1 0.5 --beta_2 0.5 --beta_1_test 0.5 --beta_2_test 0.5  \
#                --class_beta_1 0.2 --class_beta_2 0.8 --class_beta_1_test 0.2  --class_beta_2_test 0.8 \
#                 --hidden_dim 64  --train_batch_size 256 --valid_batch_size 256 &
#
#
#sleep 5
#
## bias 1
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python src/ARL.py --dataset $data --model $base_model  \
#                --epoch 200 --test_interval 1 --lr 0.0002 --beta_1 0.15 --beta_2 0.15 --beta_1_test 0.15 --beta_2_test 0.15  \
#                --hidden_dim 64  --train_batch_size 256 --valid_batch_size 256 &
#sleep 5
#
#
#
## bias 2
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python src/ARL.py --dataset $data --model $base_model  \
#                --epoch 200 --test_interval 1 --lr 0.0002 --beta_1 0.85 --beta_2 0.15 --beta_1_test 0.85 --beta_2_test 0.15  \
#                --hidden_dim 64  --train_batch_size 256 --valid_batch_size 256 &
#
#sleep 5
#
#
## bias 3
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python src/ARL.py --dataset $data --model $base_model  \
#                --epoch 200 --test_interval 1 --lr 0.0002 --beta_1 0.5 --beta_2 0.5 --beta_1_test 0.5 --beta_2_test 0.5  \
#                --class_beta_1 0.2 --class_beta_2 0.8 --class_beta_1_test 0.2  --class_beta_2_test 0.8 \
#                --hidden_dim 64  --train_batch_size 256 --valid_batch_size 256 &
#sleep 5
#
## bias 1
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python src/domind.py --dataset $data --model $base_model  \
#                --epoch 200 --test_interval 1 --lr 0.0002 --beta_1 0.15 --beta_2 0.15 --beta_1_test 0.15 --beta_2_test 0.15  \
#                --hidden_dim 64  --train_batch_size 256 --valid_batch_size 256 &
#sleep 5
#
#
#
## bias 2
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python src/domind.py --dataset $data --model $base_model  \
#                --epoch 200 --test_interval 1 --lr 0.0002 --beta_1 0.85 --beta_2 0.15 --beta_1_test 0.85 --beta_2_test 0.15  \
#                --hidden_dim 64  --train_batch_size 256 --valid_batch_size 256 &
#
#sleep 5
#
#
## bias 3
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python src/domind.py --dataset $data --model $base_model  \
#                --epoch 200 --test_interval 1 --lr 0.0002 --beta_1 0.5 --beta_2 0.5 --beta_1_test 0.5 --beta_2_test 0.5  \
#                --class_beta_1 0.2 --class_beta_2 0.8 --class_beta_1_test 0.2  --class_beta_2_test 0.8 \
#                --hidden_dim 64  --train_batch_size 256 --valid_batch_size 256 &
#sleep 5

#log=tblogging
#
#
#base_model=resnet18
#
#data=fairface
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=0 python src/main.py --dataset $data --model $base_model --log_type $log \
#              --train_batch_size 128 --valid_batch_size 128 --test_batch_size 128 \
#              --epochs 70 --test_interval 1 --lr 0.0001 --pretrained 1 --freeze_pretrain 0 --hidden_dim 128 \
#              --eps_type fair_acc --tracker_bz 50 --eps_update_interval 30 --train_eps_epochs 600 --epslr 0.00005 --reinit 1 --last_several_layers 4 &
#sleep 60
#
#data=celebA
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=2 python src/main.py --dataset $data --model $base_model --log_type $log \
#              --train_batch_size 256 --valid_batch_size 128 --test_batch_size 128 \
#              --epochs 70 --test_interval 1 --lr 0.0001 --pretrained 1 --freeze_pretrain 0 --hidden_dim 128 \
#              --eps_type fair_acc --tracker_bz 50 --eps_update_interval 30 --train_eps_epochs 600 --epslr 0.00005 --reinit 1 --last_several_layers 4 &
#sleep 60
#
#
####
#
#data=fairface
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=1 python src/main.py --dataset $data --model $base_model --log_type $log \
#              --train_batch_size 128 --valid_batch_size 128 --test_batch_size 128 \
#              --epochs 70 --test_interval 1 --lr 0.0001 --pretrained 1 --freeze_pretrain 0 --hidden_dim 128 \
#              --eps_type fair_acc --tracker_bz 20 --eps_update_interval 30 --train_eps_epochs 600 --epslr 0.00005 --reinit 1 --last_several_layers 0 &
#sleep 10
#
#data=celebA
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=3 python src/main.py --dataset $data --model $base_model --log_type $log \
#              --train_batch_size 256 --valid_batch_size 128 --test_batch_size 128 \
#              --epochs 70 --test_interval 1 --lr 0.0001 --pretrained 1 --freeze_pretrain 0 --hidden_dim 128 \
#              --eps_type fair_acc --tracker_bz 20 --eps_update_interval 30 --train_eps_epochs 600 --epslr 0.00005 --reinit 1 --last_several_layers 0
#sleep 10
#
#
#base_model=resnet50

#data=fairface
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=0 python src/main.py --dataset $data --model $base_model --log_type $log \
#              --train_batch_size 74 --valid_batch_size 74 --test_batch_size 74 \
#              --epochs 70 --test_interval 1 --lr 0.0001 --pretrained 1 --freeze_pretrain 0 --hidden_dim 128 \
#              --eps_type fair_acc --tracker_bz 50 --eps_update_interval 30 --train_eps_epochs 600 --epslr 0.00005 --reinit 1 --last_several_layers 4 &
#sleep 60
#
#data=celebA
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=2 python src/main.py --dataset $data --model $base_model --log_type $log \
#              --train_batch_size 256 --valid_batch_size 256 --test_batch_size 256 \
#              --epochs 70 --test_interval 1 --lr 0.0001 --pretrained 1 --freeze_pretrain 0 --hidden_dim 128 \
#              --eps_type fair_acc --tracker_bz 50 --eps_update_interval 30 --train_eps_epochs 600 --epslr 0.00005 --reinit 1 --last_several_layers 4 &
#sleep 60
#
#
####
#
#data=fairface
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=1 python src/main.py --dataset $data --model $base_model --log_type $log \
#              --train_batch_size 74 --valid_batch_size 74 --test_batch_size 74 \
#              --epochs 70 --test_interval 1 --lr 0.0001 --pretrained 1 --freeze_pretrain 0 --hidden_dim 128 \
#              --eps_type fair_acc --tracker_bz 10 --eps_update_interval 30 --train_eps_epochs 600 --epslr 0.00005 --reinit 1 --last_several_layers 0 &
#sleep 10
#
#data=celebA
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=3 python src/main.py --dataset $data --model $base_model --log_type $log \
#              --train_batch_size 256 --valid_batch_size 256 --test_batch_size 256 \
#              --epochs 70 --test_interval 1 --lr 0.0001 --pretrained 1 --freeze_pretrain 0 --hidden_dim 128 \
#              --eps_type fair_acc --tracker_bz 10 --eps_update_interval 30 --train_eps_epochs 600 --epslr 0.00005 --reinit 1 --last_several_layers 0 &
#sleep 10


#OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=0 python src/main.py --dataset $data --label_category $label --model $base_model --log_type $log \
#              --epochs 200 --test_interval 2 --lr 0.005 --pretrained 1 --freeze_pretrain 0 \
#              --eps_type none --tracker_bz 100 --eps_update_interval 10 --train_eps_epochs 1000 --epslr 0.001 --reinit 1 --last_several_layers 0 &
#sleep 10
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=3 python src/main.py --dataset $data --label_category $label --model $base_model --log_type $log \
#              --epochs 250 --test_interval 2 --lr 0.001 --pretrained 1 --freeze_pretrain 0 \
#              --eps_type fair --tracker_bz 100 --eps_update_interval 100 --train_eps_epochs 1000 --epslr 0.001 --reinit 1 --init_type scale --last_several_layers 0 &
#
#sleep 10
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=2 python src/main.py --dataset $data --label_category $label --model $base_model --log_type $log \
#              --epochs 250 --test_interval 2 --lr 0.001 --pretrained 1 --freeze_pretrain 0 \
#              --eps_type fair --tracker_bz 100 --eps_update_interval 100 --train_eps_epochs 1000 --epslr 0.001 --reinit 1 --init_type scale --last_several_layers 0  --accloss_type gumbel --tau 1 &
#
#sleep 10
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=2 python src/main.py --dataset $data --label_category $label --model $base_model --log_type $log \
#              --epochs 250 --test_interval 2 --lr 0.001 --pretrained 1 --freeze_pretrain 0 \
#              --eps_type fair --tracker_bz 100 --eps_update_interval 100 --train_eps_epochs 1000 --epslr 0.001 --reinit 1 --init_type scale --last_several_layers 0  --accloss_type softmax --tau 1 &
#
#sleep 10
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=2 python src/main.py --dataset $data --label_category $label --model $base_model --log_type $log \
#              --epochs 250 --test_interval 2 --lr 0.001 --pretrained 1 --freeze_pretrain 0 \
#              --eps_type fair --tracker_bz 100 --eps_update_interval 100 --train_eps_epochs 1000 --epslr 0.001 --reinit 1 --init_type randn --last_several_layers 0  --accloss_type gumbel --tau 1 &
#
#
#sleep 10
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=1 python src/main.py --dataset $data --label_category $label --model $base_model --log_type $log \
#              --epochs 250 --test_interval 2 --lr 0.001 --pretrained 1 --freeze_pretrain 0 \
#              --eps_type fair --tracker_bz 100 --eps_update_interval 100 --train_eps_epochs 1000 --epslr 0.001 --reinit 1 --init_type randn --last_several_layers 0  --accloss_type softmax --tau 1 &

#sleep 10
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=2 python src/main.py --dataset $data --label_category $label --model $base_model --log_type $log \
#              --epochs 250 --test_interval 2 --lr 0.001 --pretrained 1 --freeze_pretrain 0 \
#              --eps_type fair --tracker_bz 100 --eps_update_interval 100 --train_eps_epochs 1000 --epslr 0.001 --reinit 1 --init_type scale --last_several_layers 0  --accloss_type softmax --tau 0.5 --hard False &
#
#sleep 10
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=2 python src/main.py --dataset $data --label_category $label --model $base_model --log_type $log \
#              --epochs 250 --test_interval 2 --lr 0.001 --pretrained 1 --freeze_pretrain 0 \
#              --eps_type fair --tracker_bz 100 --eps_update_interval 100 --train_eps_epochs 1000 --epslr 0.001 --reinit 1 --init_type scale --last_several_layers 0  --accloss_type softmax --tau 1 --hard True &

#sleep 10
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=2 python src/main.py --dataset $data --label_category $label --model $base_model --log_type $log \
#              --epochs 250 --test_interval 2 --lr 0.001 --pretrained 1 --freeze_pretrain 0 \
#              --eps_type fair --tracker_bz 100 --eps_update_interval 100 --train_eps_epochs 1000 --epslr 0.001 --reinit 1 --init_type scale --last_several_layers 0  --accloss_type softmax --tau 1 --hard False &

#data=fairface
#label=age

#OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=2 python src/main.py --dataset $data --label_category $label --model $base_model --log_type $log \
#              --epochs 200 --test_interval 2 --lr 0.005 --pretrained 1 --freeze_pretrain 0 \
#              --eps_type none --tracker_bz 100 --eps_update_interval 10 --train_eps_epochs 1000 --epslr 0.001 --reinit 1 --last_several_layers 0 &
#sleep 10
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=3 python src/main.py --dataset $data --label_category $label --model $base_model --log_type $log \
#              --epochs 250 --test_interval 2 --lr 0.005 --pretrained 1 --freeze_pretrain 0 \
#              --eps_type fair --tracker_bz 100 --eps_update_interval 100 --train_eps_epochs 1000 --epslr 0.001 --reinit 1 --last_several_layers 0 &



#sleep 10
#OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=2 python src/main.py --dataset $data --label_category $label --model $base_model --log_type $log \
#              --epochs 200 --test_interval 2 --lr 0.0001 --pretrained 1 --freeze_pretrain 1 \
#              --eps_type fair --tracker_bz 100 --eps_update_interval 50 --train_eps_epochs 1000 --epslr 0.0001 --reinit 1 --last_several_layers 2 &


#base_model=cnn
#eps_type=all
#CUDA_VISIBLE_DEVICES=0 python src/main.py --dataset mnistandusps --model $base_model --log_type $log \
#                --l2_lambda 0 --epochs 500 --tracker_bz 400 --test_interval 6 --eps_update_interval 100 \
#                --train_eps_epochs 1000 --epslr 0.005  --lr 0.005 --reinit 1 --eps_type $eps_type --last_several_layers 4 &
#sleep 10
#
#
#eps_type=ce
#CUDA_VISIBLE_DEVICES=2 python src/main.py --dataset mnistandusps --model $base_model --log_type $log \
#                --l2_lambda 0 --epochs 500 --tracker_bz 400 --test_interval 6 --eps_update_interval 100 \
#                --train_eps_epochs 1000 --epslr 0.005  --lr 0.005 --reinit 1 --eps_type $eps_type --last_several_layers 4 &
#sleep 10
#
#
#eps_type=fair
#CUDA_VISIBLE_DEVICES=3 python src/main.py --dataset mnistandusps --model $base_model --log_type $log \
#                --l2_lambda 0 --epochs 500 --tracker_bz 400 --test_interval 6 --eps_update_interval 100 \
#                --train_eps_epochs 1000 --epslr 0.005  --lr 0.005 --reinit 1 --eps_type $eps_type --last_several_layers 4 &
#sleep 10



#base_model=mlp

#eps_type=none
#CUDA_VISIBLE_DEVICES=0 python src/main.py --dataset mnistandusps --model $base_model --log_type $log \
#                --l2_lambda 0 --epochs 500 --tracker_bz 400 --test_interval 6 --eps_update_interval 100 \
#                --train_eps_epochs 1000 --epslr 0.005  --lr 0.005 --reinit 1 --eps_type $eps_type --last_several_layers 0 &
#sleep 10
#
#eps_type=all
#CUDA_VISIBLE_DEVICES=0 python src/main.py --dataset mnistandusps --model $base_model --log_type $log \
#                --l2_lambda 0 --epochs 500 --tracker_bz 400 --test_interval 6 --eps_update_interval 100 \
#                --train_eps_epochs 1000 --epslr 0.005  --lr 0.005 --reinit 1 --eps_type $eps_type --last_several_layers 0 &
#sleep 10


#eps_type=ce
#CUDA_VISIBLE_DEVICES=2 python src/main.py --dataset mnistandusps --model $base_model --log_type $log \
#                --epochs 500 --lr 0.005 --tracker_bz 400 --test_interval 6 --eps_update_interval 100 \
#                --train_eps_epochs 1000 --epslr 0.005  --reinit 1 --eps_type $eps_type --last_several_layers 0 &
#sleep 10


#eps_type=fair
#CUDA_VISIBLE_DEVICES=3 python src/main.py --dataset mnistandusps --model $base_model --log_type $log \
#                --l2_lambda 0 --epochs 500 --tracker_bz 400 --test_interval 6 --eps_update_interval 100 \
#                --train_eps_epochs 1000 --epslr 0.005  --lr 0.005 --reinit 1 --eps_type $eps_type --last_several_layers 0 &
#sleep 10
#
#
#eps_type=fair
#CUDA_VISIBLE_DEVICES=0 python src/main.py --dataset mnistandusps --model $base_model --log_type $log \
#                --l2_lambda 0 --epochs 500 --tracker_bz 400 --test_interval 6 --eps_update_interval 100 \
#                --train_eps_epochs 1000 --epslr 0.005  --lr 0.005 --reinit 1 --eps_type $eps_type --last_several_layers 2 &
#sleep 10



