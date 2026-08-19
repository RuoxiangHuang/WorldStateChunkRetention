# Matrix-Game 2.0 World-State CR

Host transfer of LingBot **World-State CR** onto Skywork **Matrix-Game 2.0**.
This tree is **not** `lingbot-world/` (that remains the LingBot source of truth).

Code: [`matrix-game-2/`](../../matrix-game-2/)  
Detail: [`matrix-game-2/docs/CR_TRANSFER.md`](../../matrix-game-2/docs/CR_TRANSFER.md)

## What was added

- `wan/memory/kv_retention.py` — sink / ranked archive / recent on the rolling self-attn KV (`local_attn_size=6`)
- `--memory-policy {window,world_state_cr}` (default `window` = official FIFO)
- Action-module KV stays FIFO; selector checkpoints are **not** loaded

This branch is **CR only**. TICH / VAE stream opts / FP8 FFN are not included.

## Run

```bash
cd matrix-game-2
python wan/memory/test_kv_retention.py
python inference.py \
    --config_path configs/inference_yaml/inference_universal.yaml \
    --checkpoint_path <ckpt> \
    --pretrained_model_path <weights> \
    --memory-policy world_state_cr \
    --cr-sink-frames 1 --cr-recent-frames 1
```

Weights are not in this repo. Window size is unchanged, so this is a **memory-axis** method (which frames stay), not a speedup.
