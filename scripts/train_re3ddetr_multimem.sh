python main.py \
    --run_name redetr_multimem \
    --checkpoint redetr_multimem.pt \
    --bootstrap analogical_multimem_within.pt \
    --checkpoint_path CHECKPOINT_PATH \
    --anno_path PATH_TO_PARTNET \
    --feat_path FEAT_PATH \
    --label_path PATH_TO_PARTNET_REPO/stats/after_merging_label_ids/ \
    --retriever_train_mode canonical --retriever_val_mode canonical \
    --retriever_ckpt analogical_within.pt \
    --lr 1e-4 --batch_size 7 --epochs 400 \
    --model analogical_nets --no_mem_decoding \
    --rotary_pe --pre_norm --feat_dim 264 \
    --queries 64 --negative_obj_weight 0.2 \
    --num_memories 5 --same_wild_augment_train \
    --cross_instance --hungarian \
    --train_top_mem_pool_size 20 --val_top_mem_pool_size 1 \
    --train_split multicat12 --val_with_semantics