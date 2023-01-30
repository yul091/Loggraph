
# Generate torch_geometric dataset

########################################## DATASET ###########################################
DATA_NAME=sockshop # AIT, BGL, sockshop
INTERVAL=3s # 3s, 5s, 10s, 0.5min, 1min, 2min, 5min, 10min, 15min, 30min, 1h

INFERENCE=regex # regex, seq2seq
ROOT=dataset/${DATA_NAME}/${INFERENCE}-node-${INTERVAL}-template-bertembed
# DATA=dataset/BGL/BGL.log_structured.csv
# DATA=dataset/AIT-LDS-v1_1/processed/whole-dataset-processed.json
DATA=dataset/sockshop/database_customer_v3.csv
STRATEGY=0 # 0, 1
NUM_SHOTS=10 # 10, 5
NEG_RATE=1.5 # 1.5 for AIT dataset
MODEL_PATH=results/prompt/AIT/${NUM_SHOTS}-shot-${STRATEGY}-${NEG_RATE}neg
###############################################################################################

CUDA_VISIBLE_DEVICES=7 python graph_generation.py \
    --root ${ROOT} \
    --log_file ${DATA} \
    --inference_type ${INFERENCE} \
    --strategy ${STRATEGY} \
    --label_type node \
    --pretrained_model_name_or_path ${MODEL_PATH} \
    --interval ${INTERVAL} \
    --event_template 



# for INTERVAL in 1min 2min 5min 10min 15min 30min 1h
# do
#     INFERENCE=seq2seq # OR: regex, seq2seq
#     ROOT=dataset/${DATA_NAME}/${INFERENCE}-node-${INTERVAL}-template-bertembed
#     DATA=dataset/BGL/BGL.log_structured.csv
#     # DATA=dataset/AIT-LDS-v1_1/processed/whole-dataset-processed.json
#     STRATEGY=0 # OR: 1
#     NUM_SHOTS=10 # OR: 5
#     NEG_RATE=1.5 # 1.5 for AIT dataset
#     MODEL_PATH=results/prompt/AIT/${NUM_SHOTS}-shot-${STRATEGY}-${NEG_RATE}neg
#     ###############################################################################################

#     CUDA_VISIBLE_DEVICES=1 python graph_generation.py \
#         --root ${ROOT} \
#         --log_file ${DATA} \
#         --inference_type ${INFERENCE} \
#         --strategy ${STRATEGY} \
#         --label_type node \
#         --pretrained_model_name_or_path ${MODEL_PATH} \
#         --interval ${INTERVAL} \
#         --event_template 

# done