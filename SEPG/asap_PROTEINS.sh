read -p "Gpu:" gpu
echo "Input seed:"
read -a array
for i in ${array[@]}
do
    python trainer_sep_args.py --dataset PROTEINS \
                               --hidden_dim 64 \
                               --batch_size 128 \
                               --final_dropout 0 \
                               --seed ${i} \
                               --gpu ${gpu} \
                               --rerun_under_asap_setting
done 