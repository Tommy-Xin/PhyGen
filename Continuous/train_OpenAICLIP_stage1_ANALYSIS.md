# train_OpenAICLIP_stage1.py 详细分析

## 📋 文件概述

这是 GenHancer Continuous 版本的 Stage 1 训练脚本，用于训练轻量级 denoiser 和 CLIP 投影层。该脚本实现了基于 Rectified Flow 的连续扩散模型训练。

---

## 🏗️ 模型架构

### SuperModel 结构
```python
SuperModel = {
    clip_vis: OpenAICLIP (CLIP视觉编码器 + 投影层)
    dit: Flux (轻量级DiT denoiser)
}
```

### 组件说明

1. **CLIP视觉编码器 (clip_vis)**
   - 基础模型：OpenAI CLIP ViT-L/14 (224px 或 336px)
   - 投影层：
     - `project_clip`: 768 → clip_dim (768) → clip_dim (768)
     - `project_t5`: 768 → t5_dim (4096) → t5_dim (4096)
   - **训练状态**：仅训练 `project_clip` 和 `project_t5`，CLIP 主体冻结

2. **DiT Denoiser (dit)**
   - 架构：轻量级 Flux-like DiT
   - 配置：`depth=2` (双块), `depth_single_blocks=4` (单块)
   - **训练状态**：完全可训练，使用 bfloat16 精度

3. **VAE**
   - 用于图像编码/解码
   - **训练状态**：完全冻结

---

## 🔄 训练流程

### 1. 初始化阶段 (Lines 90-152)

```python
# 加载配置
args = OmegaConf.load(parse_args())

# 初始化 Accelerator (支持多卡训练)
accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision,  # bf16
    log_with=args.report_to,  # tensorboard
)

# 加载模型组件
dit = load_flow_model2(args.model_name, device="cpu")  # 随机初始化
vae = load_ae(args.model_name, device=accelerator.device)  # 预训练权重
clip_vis = load_clip_model_OpenAICLIP(args.clip_config, device=accelerator.device)  # 预训练权重

# 设置训练状态
vae.requires_grad_(False)  # VAE冻结
dit.requires_grad_(True)   # DiT可训练
clip_vis: 仅 project_clip 和 project_t5 可训练
```

### 2. 数据准备 (Lines 244-258)

```python
# 1. 图像编码
x_1 = vae.encode(NORMALIZE_VAE(original_img))  # [B, 16, H/8, W/8]

# 2. CLIP特征提取
inp = prepare_clip(
    clip=super_model.clip_vis,
    original_img=NORMALIZE_CLIP(original_img),  # CLIP归一化
    img=x_1  # VAE latent
)
# 返回: {img, img_ids, txt, txt_ids, vec}

# 3. Rectified Flow 插值
t = torch.sigmoid(torch.randn((bs,)) * args.scale_factor)  # 随机时间步
x_0 = torch.randn_like(x_1)  # 噪声
x_t = (1 - t) * x_1 + t * x_0  # 线性插值
```

### 3. 前向传播 (Lines 261-267)

```python
model_pred = super_model.dit(
    img=x_t,           # 插值后的latent [B, H*W, 64]
    img_ids=inp['img_ids'],  # 位置编码
    txt=inp['txt'],     # T5投影特征 [B, 1, 4096]
    txt_ids=inp['txt_ids'],
    y=inp['vec'],       # CLIP投影特征 [B, 768]
    timesteps=t,        # 时间步 [B]
    guidance=guidance_vec,  # CFG=4.0
)
```

### 4. 损失计算 (Line 269)

```python
loss = F.mse_loss(
    model_pred.float(), 
    (x_0 - x_1).float(),  # 目标：速度场 (x_0 - x_1)
    reduction="mean"
)
```

**损失含义**：预测从 `x_t` 到 `x_0` 的速度场，这是 Rectified Flow 的核心。

### 5. 反向传播与优化 (Lines 276-281)

```python
accelerator.backward(loss)
accelerator.clip_grad_norm_(super_model.parameters(), args.max_grad_norm)
optimizer.step()
lr_scheduler.step()
optimizer.zero_grad()
```

---

## 📊 关键参数分析

### 训练参数

| 参数 | 默认值 | 说明 | 多卡影响 |
|------|--------|------|----------|
| `train_batch_size` | 16 | 每卡batch size | **需要调整** |
| `gradient_accumulation_steps` | 4 | 梯度累积步数 | **需要调整** |
| `learning_rate` | 1e-4 | 学习率 | 保持不变 |
| `max_train_steps` | 100000 | 最大训练步数 | 保持不变 |
| `checkpointing_steps` | 50000 | 检查点保存间隔 | 保持不变 |
| `mixed_precision` | "bf16" | 混合精度 | 保持不变 |
| `max_grad_norm` | 1.0 | 梯度裁剪 | 保持不变 |

### 模型参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `model_name` | "flux-dev" | DiT配置名称 |
| `clip_image_size` | 224/336 | CLIP输入尺寸 |
| `clip_dim` | 768 | CLIP投影维度 |
| `t5_dim` | 4096 | T5投影维度 |
| `scale_factor` | 1.0 | 时间步采样缩放因子 |

### 数据参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `img_size` | 224 | 训练图像尺寸 |
| `num_workers` | 4 | 数据加载进程数 |
| `patch_size` | 1 | 补丁大小 |

---

## 🔧 多卡训练参数调整指南

### 从 8 卡 → 4 卡需要修改的参数

#### 1. **必须修改的参数**

**accelerate_config.yaml:**
```yaml
num_processes: 4  # 8 → 4
```

**训练脚本 (CUDA_VISIBLE_DEVICES):**
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 指定4张卡
```

#### 2. **保持相同有效batch size的调整**

**当前配置 (8卡):**
- `train_batch_size = 16`
- `gradient_accumulation_steps = 4`
- `num_processes = 8`
- **有效batch size = 16 × 8 × 4 = 512**

**4卡配置选项:**

**选项A: 增加每卡batch size**
```yaml
train_batch_size: 32  # 16 → 32 (翻倍)
gradient_accumulation_steps: 4  # 保持不变
# 有效batch size = 32 × 4 × 4 = 512 ✓
```

**选项B: 增加梯度累积**
```yaml
train_batch_size: 16  # 保持不变
gradient_accumulation_steps: 8  # 4 → 8 (翻倍)
# 有效batch size = 16 × 4 × 8 = 512 ✓
```

**选项C: 同时调整**
```yaml
train_batch_size: 24  # 16 → 24
gradient_accumulation_steps: 5  # 4 → 5
# 有效batch size = 24 × 4 × 5 = 480 (接近512)
```

#### 3. **学习率调整建议**

- **如果保持相同有效batch size**: 学习率保持不变 (`1e-4`)
- **如果batch size减半**: 学习率减半 (`5e-5`)

#### 4. **其他不需要修改的参数**

- `learning_rate`: 保持不变
- `max_train_steps`: 保持不变
- `checkpointing_steps`: 保持不变
- `lr_warmup_steps`: 保持不变 (但实际warmup步数会因 `accelerator.num_processes` 自动调整)
- `num_workers`: 可以保持不变或适当减少

---

## 💾 检查点保存机制

### 保存的组件 (Lines 290-306)

1. **checkpoint-dit-{step}.bin**: DiT denoiser权重
2. **checkpoint-project-clip-{step}.bin**: CLIP投影层权重
3. **checkpoint-project-t5-{step}.bin**: T5投影层权重
4. **optimizer-state-{step}.bin**: 优化器状态

### 保存时机

- 每 `checkpointing_steps` 步保存一次 (默认50000步)
- 训练结束时保存最终权重

### 恢复训练

```python
resume_from_checkpoint: latest  # 自动加载最新checkpoint
```

---

## 🎯 训练目标

### Stage 1 训练目标

1. **训练轻量级denoiser (DiT)**
   - 学习从噪声到图像的映射
   - 使用CLIP特征作为条件

2. **训练CLIP投影层**
   - `project_clip`: 将CLIP特征投影到denoiser输入空间
   - `project_t5`: 将CLIP特征投影到T5-like条件空间

3. **冻结组件**
   - CLIP主体：保持预训练特征
   - VAE：保持预训练编码/解码能力

---

## 🔍 关键代码片段解析

### 1. Rectified Flow 插值 (Lines 254-256)

```python
t = torch.sigmoid(torch.randn((bs,)) * args.scale_factor)
x_0 = torch.randn_like(x_1)
x_t = (1 - t[:, None, None]) * x_1 + t[:, None, None] * x_0
```

- `t ∈ [0, 1]`: 随机时间步（使用sigmoid确保在[0,1]区间）
- `x_1`: 真实图像latent
- `x_0`: 随机噪声
- `x_t`: 插值后的latent

### 2. 损失函数 (Line 269)

```python
loss = F.mse_loss(model_pred.float(), (x_0 - x_1).float(), reduction="mean")
```

**Rectified Flow 损失**：预测速度场 `v = x_0 - x_1`，而不是预测噪声。

### 3. CLIP特征准备 (Line 251)

```python
inp = prepare_clip(
    clip=super_model.clip_vis,
    original_img=NORMALIZE_CLIP(original_img).to(weight_dtype),
    img=x_1.to(weight_dtype)
)
```

- `original_img`: 原始图像（CLIP归一化）
- `img`: VAE编码后的latent
- 返回CLIP投影特征和位置编码

### 4. 梯度累积 (Line 244)

```python
with accelerator.accumulate(super_model):
    # 训练代码
```

确保梯度累积在多卡训练中正确工作。

---

## ⚠️ 注意事项

1. **内存优化**
   - DiT使用 `bfloat16` 精度
   - VAE编码使用 `float32`（避免精度损失）
   - CLIP使用 `float32`

2. **336px特殊处理** (Lines 132-134)
   ```python
   if args.clip_config.clip_image_size == 336:
       clip_vis.model.visual_projection.weight = torch.nn.Parameter(...)
   ```
   确保投影权重连续存储，避免性能问题。

3. **T5序列长度** (Line 93)
   ```python
   args.clip_config.seq_t5 = 256 if is_schnell else 512
   ```
   根据模型类型调整T5序列长度。

4. **数据集大小** (Line 165)
   ```python
   num_update_steps_per_epoch = math.ceil(int(3e6) / args.data_config.train_batch_size)
   ```
   假设CC3M数据集约300万张图像。

---

## 📈 训练监控

### 日志记录

- **TensorBoard**: `accelerator.log({"train_loss": train_loss}, step=global_step)`
- **进度条**: 显示当前loss和学习率
- **检查点**: 每50000步保存一次

### 关键指标

- `train_loss`: MSE损失（速度场预测误差）
- `lr`: 当前学习率
- `global_step`: 全局训练步数

---

## 🔗 相关文件

- **配置文件**: `train_configs/test_OpenAICLIP_224_stage1.yaml`
- **加速配置**: `train_configs/accelerate_config.yaml`
- **启动脚本**: `train_scripts/scripts_train_OpenAICLIP_224_stage1.sh`
- **Stage 2训练**: `train_OpenAICLIP_stage2_all.py`
- **模型定义**: `src/flux/util.py`, `clip_models/build_CLIP.py`

---

## 📝 总结

这个训练脚本实现了 GenHancer Stage 1 的完整训练流程：

1. ✅ 加载并配置模型组件（CLIP、DiT、VAE）
2. ✅ 实现 Rectified Flow 训练流程
3. ✅ 支持多卡分布式训练（DeepSpeed ZeRO-2）
4. ✅ 混合精度训练（bf16）
5. ✅ 梯度累积和梯度裁剪
6. ✅ 检查点保存和恢复
7. ✅ 训练监控和日志记录

**关键特点**：轻量级denoiser + CLIP投影层联合训练，为Stage 2的CLIP微调做准备。



