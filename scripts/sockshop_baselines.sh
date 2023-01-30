# Dynamic transformer + GCNAE
DATANAME=sockshop # AIT, BGL, sockshop
LABEL_TECH=regex # seq2seq, regex
INTERVAL=3s # 3s, 5s, 10s, 0.5min, 1min, 2min, 5min, 10min, 30min
TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=32
WEIGHT_DECAY=5e-7
MAX_LENGTH=1024
LAMBDA=0.1
LAYERS=2
CLASSIFICATION=edge
GLOBAL_WEIGHT=1
LR=1e-3
ROOT=dataset/${DATANAME}/${LABEL_TECH}-node-${INTERVAL}-template-bertembed
MODEL_PATH=bert-base-uncased # facebook/bart-base, gpt2, xlnet-base-cased, bert-base-uncased, bert-base-cased


for MODEL_TYPE in ae-gcnae ae-mlpae ae-dominant ae-conad ae-anomalydae deeptralog addgraph
do
    CKPT=results/${DATANAME}/${LABEL_TECH}-${CLASSIFICATION}/${INTERVAL}/${MODEL_TYPE} # gcn, mlp, etc.
    CUDA_VISIBLE_DEVICES=6 python main.py \
        --root ${ROOT} \
        --checkpoint_dir ${CKPT} \
        --train_batch_size ${TRAIN_BATCH_SIZE} \
        --eval_batch_size ${EVAL_BATCH_SIZE} \
        --model_type ${MODEL_TYPE} \
        --pretrained_model_path ${MODEL_PATH} \
        --lambda_seq ${LAMBDA} \
        --classification ${CLASSIFICATION} \
        --max_length ${MAX_LENGTH} \
        --lr ${LR} \
        --layers ${LAYERS} \
        --weight_decay ${WEIGHT_DECAY} \
        --do_train \
        --do_eval \
        --multi_granularity \
        --global_weight ${GLOBAL_WEIGHT} \
        --from_scratch
done