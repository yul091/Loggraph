DATA_NAME=AIT
INFERENCE=seq2seq # regex, seq2seq
cd dataset/${DATA_NAME}

for INTERVAL in 2min 5min 10min 15min 30min 1h
do
    ROOT=${INFERENCE}-node-${INTERVAL}-template-bertembed
    unzip ${ROOT}.zip

done