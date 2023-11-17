OUT_DIR = "/Your/output/path"
DATA_PATH = "/Your/data/path"
DATA_NAME = "xsum_data"
PRETRAIN_CKPT_PATH = "/Your/pretrain_ckpt/path"


python -u -m torch.distributed.launch --nproc_per_node=4 --master_port=9421 \
Genie_Finetune.py
--checkpoint_path="/save/delibot" \
--model_channels 128 --in_channel 128 --out_channel 128 --vocab_size 30522 \
--config_name="bert-base-uncased" --token_emb_type="random" --model_arch="s2s_CAT" \
--diffusion_steps 2000 --predict_xstart --noise_schedule="sqrt" --training_mode="s2s" \
--schedule_sampler="loss-second-moment" --tgt_max_len 64 --src_max_len 512 --data_name="deli_data_small" \
--data_path="data" \
--lr_anneal_steps 120000 --batch_size 64 --lr 5e-05 --warmup_steps 7200 --train_type="S2S_Diffusion" \
--eval_interval 200 --log_interval 200 --save_interva 20000 \
--pretrain_model_path="/save/GENIE_ckpt-500w"