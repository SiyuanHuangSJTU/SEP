array_num_blocks=(2 4)
array_tree_depth=(3 5)
array_hidden_dim=(32 128)
array_conv_dropout=(0 0.5)
array_pooling_dropout=(0)
array_l2rate=(0.0005 0.02)
array_splits=(0 1 2 3 4 5 6 7 8 9)

for nb in ${array_num_blocks[@]}
do
for td in ${array_tree_depth[@]}
do
for hd in ${array_hidden_dim[@]}
do
for cd in ${array_conv_dropout[@]}
do
for pd in ${array_pooling_dropout[@]}
do
for l2r in ${array_l2rate[@]}
do
for split in ${array_splits[@]}
do
    python trainer_sepu_args.py --dataset Citeseer \
                                --num_blocks ${nb} \
                               --tree_depth ${td} \
                               --hidden_dim ${hd} \
                               --conv_dropout ${cd} \
                               --pooling_dropout ${pd} \
                               --l2rate ${l2r} \
                               --index_split ${split}
done
done
done
done
done
done
done