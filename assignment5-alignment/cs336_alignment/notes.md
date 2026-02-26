## vLLM 对推理速度的影响
1. 不使用vLLM：
    - 时间：15.3513 seconds
2. 使用vLLM：
    - 时间：0.2849 seconds

## MATH 数据集
### 下载
```bash
# 创建对应目录
cd cs336_alignments
mkdir -p data/math

# 下载数据集
hfd garg-aayush/sft-cs336-assign5-datasets --local-dir ./data/math/raw/ --dataset

# 将数据集移动到指定位置
ln -sf ./raw/sft-reason/val.jsonl data/math/validation.jsonl
ln -sf ./raw/sft-reason/sft_gpt-oss-120b_filtered.jsonl data/math/sft.jsonl
```
### 处理
1. 调整编码为可读 UTF-8
2. 修改为标准 JSONL 格式
