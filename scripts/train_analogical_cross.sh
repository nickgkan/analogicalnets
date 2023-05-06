python main.py \
    --run_name analogical_cross_identity \
    --checkpoint analogical_cross.pt \
    --bootstrap analogical_within.pt \
    --checkpoint_path CHECKPOINT_PATH \
    --anno_path PATH_TO_PARTNET \
    --feat_path FEAT_PATH \
    --label_path PATH_TO_PARTNET_REPO/stats/after_merging_label_ids/ \
    --retriever_train_mode canonical_class_based --retriever_val_mode canonical \
    --retriever_ckpt analogical_within.pt \
    --lr 2e-4 --batch_size 16 --epochs 400 \
    --model analogical_nets \
    --rotary_pe --pre_norm --feat_dim 264 \
    --queries 64 --negative_obj_weight 0.2 \
    --num_memories 1 --same_wild_augment_train \
    --cross_instance --hungarian \
    --train_top_mem_pool_size 20 --val_top_mem_pool_size 1 \
    --train_split multicat12 --val_with_semantics \
    # --k_shot 5 --eval_multitask 10 --eval \
    # --k_shot_seed 10 \
    # --ft_epoch 90  # uncomment this if you want to fine-tune
