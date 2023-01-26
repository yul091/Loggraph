# Label entities using BART-based seq2seq learning
DATA_NAME=AIT
DATA=dataset/dataset.json

################################## For prompt labeling ######################################
LABEL_METHOD=prompt # OR: prompt, regex
NUM_SHOTS=10 # OR: 5
STRATEGY=0
EPOCHS=30
SEED=2019
N_GRAMS=8 # 8 for AIT dataset, 5 for BGL dataset
NEG_RATE=1.5 # 1.5 for AIT dataset, 10 for BGL dataset
TRAIN_BATCH=3
EVAL_BATCH=8
GRAD_CUM_STEPS=16
PRETRAINED_MODEL=facebook/bart-large
OUT_DIR=dataset/NER/${DATA_NAME}-${N_GRAMS}grams-${NEG_RATE}neg
CKPT_DIR=results/BART_seq2seq/${DATA_NAME}/${NUM_SHOTS}-shot-${STRATEGY}-${NEG_RATE}neg
#############################################################################################

################################### For regex labeling ######################################
# LABEL_METHOD=regex # OR: prompt, regex
# NUM_SHOTS=10 # OR: 5
# STRATEGY=0
# EPOCHS=30
# SEED=2019
# N_GRAMS=5 # 8 for AIT dataset, 5 for BGL dataset
# NEG_RATE=10 # 1.5 for AIT dataset, 10 for BGL dataset
# TRAIN_BATCH=5
# EVAL_BATCH=5
# GRAD_CUM_STEPS=16
# PRETRAINED_MODEL=facebook/bart-large
# OUT_DIR=/nfs/intern_data/yufli/dataset/NER/BGL-regex
# CKPT_DIR=results/BART_seq2seq/${DATA_NAME}/regex
#############################################################################################


CUDA_VISIBLE_DEVICES=0 python NER.py \
    --gen_data ${DATA} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUT_DIR} \
    --strategy ${STRATEGY} \
    --n_shots ${NUM_SHOTS} \
    --n_grams ${N_GRAMS} \
    --neg_rate ${NEG_RATE} \
    --labeling_technique ${LABEL_METHOD} \
    --model_name_or_path ${PRETRAINED_MODEL} \
    --num_train_epochs ${EPOCHS} \
    --do_train \
    --do_eval \
    --train_batch_size ${TRAIN_BATCH} \
    --eval_batch_size ${EVAL_BATCH} \
    --gradient_accumulation_steps ${GRAD_CUM_STEPS} \
    --ckpt_dir ${CKPT_DIR} \
    --seed ${SEED} \
    --overwrite_cache