# GPU配置说明

## 指定可用GPU

在GenHancer Continuous训练中，有两种方式指定可用的GPU：

### 方法1：在启动脚本中设置（推荐）

在训练脚本（如 `train_scripts/scripts_train_OpenAICLIP_224_stage1.sh`）中添加：

```bash
# 指定使用GPU 0, 1, 2, 3
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 或者只使用GPU 0和1
export CUDA_VISIBLE_DEVICES=0,1
```

**重要**：设置 `CUDA_VISIBLE_DEVICES` 后，需要同步修改 `accelerate_config.yaml` 中的 `num_processes`：

```yaml
num_processes: 4  # 必须等于CUDA_VISIBLE_DEVICES中指定的GPU数量
```

### 方法2：在命令行中直接指定

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file "train_configs/accelerate_config.yaml" train_OpenAICLIP_stage1.py --config "train_configs/test_OpenAICLIP_224_stage1.yaml"
```

## 配置步骤

### 步骤1：修改启动脚本

编辑 `train_scripts/scripts_train_OpenAICLIP_224_stage1.sh`：

```bash
export AE="/home/gaiyiming/hjq/xinc/DiffusionCLIP/pretrained_weights/FLUX.1-dev/ae.safetensors"

# 指定可用的GPU（例如：使用GPU 0,1,2,3）
export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch --config_file "train_configs/accelerate_config.yaml" train_OpenAICLIP_stage1.py --config "train_configs/test_OpenAICLIP_224_stage1.yaml"
```

### 步骤2：修改accelerate配置

编辑 `train_configs/accelerate_config.yaml`：

```yaml
num_processes: 4  # 必须等于CUDA_VISIBLE_DEVICES中指定的GPU数量
```

## 示例配置

### 使用4个GPU（0,1,2,3）

**启动脚本**：
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

**accelerate_config.yaml**：
```yaml
num_processes: 4
```

### 使用2个GPU（0,1）

**启动脚本**：
```bash
export CUDA_VISIBLE_DEVICES=0,1
```

**accelerate_config.yaml**：
```yaml
num_processes: 2
```

### 使用单个GPU（0）

**启动脚本**：
```bash
export CUDA_VISIBLE_DEVICES=0
```

**accelerate_config.yaml**：
```yaml
num_processes: 1
```

## 注意事项

1. **num_processes必须匹配**：`num_processes` 必须等于 `CUDA_VISIBLE_DEVICES` 中指定的GPU数量
2. **GPU编号**：`CUDA_VISIBLE_DEVICES` 中的编号是物理GPU编号（从0开始）
3. **DeepSpeed配置**：如果使用DeepSpeed，确保 `zero_stage` 和 `gradient_accumulation_steps` 配置合理

## 验证GPU设置

运行训练前，可以验证GPU是否被正确识别：

```bash
# 查看所有GPU
nvidia-smi

# 测试CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0,1 python -c "import torch; print(torch.cuda.device_count())"
# 应该输出: 2
```










