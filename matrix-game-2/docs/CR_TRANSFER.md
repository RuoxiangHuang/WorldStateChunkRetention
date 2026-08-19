# LingBot CR → Matrix-Game 2.0

## 结论

**可以迁完整 KV-CR**，挂在 `CausalWanSelfAttention` 的 overflow 驱逐上。  
默认 `--memory-policy window` = 官方 FIFO，行为不变。

本目录只含这条记忆轴。不含 TICH / VAE Graph / FP8 FFN。

## 架构

| | LingBot | Matrix-Game 2.0 |
|--|--|--|
| 长程记忆 | 跨 chunk DiT KV archive | rolling self-attn KV，`local_attn_size=6` |
| 官方驱逐 | window / CR | FIFO，`sink_size=0` |
| 本次 CR | sink / recent / archive 重排 buffer | 同左，token 级 gather |
| 动作 KV | 无 | mouse/keyboard cache **仍 FIFO** |
| selector | `selector_ws_*.pt` | **不加载** |

满窗时官方丢掉最老 3 帧。`world_state_cr` 在同样 6 帧预算里改成：

`[sink 首帧 | 按 mouse 积分位姿排序的 archive | recent] + 新 block`

attention 仍吃连续 `local_end - window : local_end`，不改 flash-attn。

## 怎么跑

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

## 不要承诺的

- 训练只看最近 6 帧，把更早的 sink/archive 塞回去是 OOD，回环可能变好也可能糊
- 不会自动有 LingBot 同比例加速；窗口大小没变，变的是窗口里是谁
