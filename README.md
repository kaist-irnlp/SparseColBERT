# TPU-cont: Continual Training in TPU

This branch is for trianing continually in colab TPU.

First, you need to install HuggingFace:
```
!pip install transformers
```

After installing transformers, you should change the branch to TPU-cont and mv "trainer.py" in this repo to HuggingFace library directory installed in your system.
```
%cd /content/SparseColBERT 
!git checkout TPU-cont
!mv trainer.py /usr/local/lib/python3.6/dist-packages/transformers/
```

Then, run the training code with arguments as follows:

```
PREV_STEPS = 0 # 0, 12500, 25000, ..., 87500
MAX_STEPS = 100000 # fixed for scheduler
SAVE_STEPS = int(MAX_STEPS / 8) # 12500 steps 
NUM_CORES = 8
BATCH_SIZE = 16
N = 8192
K = 0.005
TRAINING_INS_NUM = SAVE_STEPS * BATCH_SIZE * NUM_CORES
TRAINING_INS_START_FROM = PREV_STEPS * BATCH_SIZE * NUM_CORES
if PREV_STEPS == 0:
    PREV_CHECKPOINT = None
else:
    PREV_CHECKPOINT = "your output directory/checkpoint-" + str(PREV_STEPS)

!python -m src.xla_spawn --num_cores $NUM_CORES \
	src/train.py \
	--triples triples.train.small.tsv \
  --data_dir "your data directory" \
  --maxsteps $MAX_STEPS \
  --output_dir "your output directory" \
  --per_device_train_batch_size $BATCH_SIZE \
  --training_ins_num $TRAINING_INS_NUM \
  --training_is_start_from $TRAINING_INS_START_FROM \
  --num_train_epochs 1 \
  --save_steps $SAVE_STEPS \
  --prev_checkpoint $PREV_CHECKPOINT \
  --n $N \
  --k $K \
```
