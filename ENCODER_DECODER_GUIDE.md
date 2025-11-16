# Encoder-Decoder Transformer Guide

이 가이드는 새로 추가된 **Encoder-Decoder Transformer** (원본 Transformer 아키텍처) 사용 방법을 설명합니다.

## 프로젝트에 두 가지 Transformer가 있습니다

### 1. Decoder-only (GPT-like) - 기존
- **파일**: `models/transformer.py`
- **구조**: Decoder만
- **용도**: 언어 모델링, 텍스트 생성
- **학습**: `train_ddp.py`
- **Inference**: `inference.py`

### 2. Encoder-Decoder (Original Transformer) - 새로 추가
- **파일**: `models/transformer_enc_dec.py`
- **구조**: Encoder + Decoder
- **용도**: Seq2Seq (번역, 요약 등)
- **학습**: `train_enc_dec_ddp.py`
- **Inference**: `inference_enc_dec.py`

## Encoder-Decoder 아키텍처

```
Source Sequence                Target Sequence
      ↓                              ↓
  Embedding                      Embedding
      ↓                              ↓
┌─────────────┐                ┌─────────────┐
│  ENCODER    │                │  DECODER    │
│             │                │             │
│ Self-Attn   │   ─────────→   │ Self-Attn   │ (causal)
│     +       │   Encoder Out  │     +       │
│ Feed-Fwd    │                │ Cross-Attn  │ ← Encoder 결과 참조
│             │                │     +       │
│  × N layers │                │ Feed-Fwd    │
└─────────────┘                └─────────────┘
                                     ↓
                                  Output
```

### 주요 특징

1. **Encoder**
   - Bidirectional self-attention (양방향)
   - Causal mask 없음 (전체 입력 볼 수 있음)
   - Source sequence를 이해

2. **Decoder**
   - Causal self-attention (단방향)
   - Cross-attention (Encoder 출력 참조)
   - Target sequence를 생성

3. **Cross-Attention**
   - Decoder가 Encoder 출력을 참조
   - Query: Decoder, Key/Value: Encoder
   - 번역/요약 등에 필수

## 빠른 시작

### 1. Seq2Seq Dataset 준비

현재 3가지 toy task 지원:

#### Reversal (기본)
```python
# Input:  "hello"
# Output: "olleh"
```

#### Copy
```python
# Input:  "hello"
# Output: "hello"
```

#### Addition
```python
# Input:  "12+34"
# Output: "46"
```

### 2. 학습 (8 GPU)

```bash
# Reversal task
torchrun --nproc_per_node=8 train_enc_dec_ddp.py --config configs/enc_dec_config.yaml

# 또는 스크립트 사용
bash scripts/train_enc_dec_8gpu.sh
```

### 3. Inference

```bash
# Interactive mode
python inference_enc_dec.py --checkpoint checkpoints_enc_dec/best_model.pt --interactive

# Single translation
python inference_enc_dec.py \
    --checkpoint checkpoints_enc_dec/best_model.pt \
    --source "hello world" \
    --max-length 50
```

## 설정

`configs/enc_dec_config.yaml` 수정:

```yaml
model:
  dim: 256        # Model dimension
  depth: 6        # Number of encoder/decoder layers
  heads: 8        # Attention heads

training:
  batch_size: 64  # Per GPU
  num_epochs: 50

data:
  task: "reversal"  # "reversal", "copy", or "addition"
  num_train_samples: 10000
  num_val_samples: 1000
```

## Task별 예상 성능

### Reversal
- **Epochs**: 20-30
- **Val Accuracy**: >95%
- **예시**:
  - Input: "transformer"
  - Output: "remrofsnart"

### Copy
- **Epochs**: 10-20
- **Val Accuracy**: >98%
- **예시**:
  - Input: "attention"
  - Output: "attention"

### Addition
- **Epochs**: 30-50
- **Val Accuracy**: >90%
- **예시**:
  - Input: "25+37"
  - Output: "62"

## Databricks에서 실행

### 노트북 셀

```python
# 1. GPU 확인
import torch
print(f"GPUs: {torch.cuda.device_count()}")

# 2. 학습
!torchrun --nproc_per_node=8 train_enc_dec_ddp.py --config configs/enc_dec_config.yaml

# 3. Inference
!python inference_enc_dec.py --checkpoint checkpoints_enc_dec/best_model.pt --source "hello" --max-length 20
```

## 커스텀 Dataset

자신만의 Seq2Seq 데이터를 사용하려면 `data/seq2seq_dataset.py`를 수정:

```python
class CustomSeq2SeqDataset(Dataset):
    def __init__(self, source_file, target_file):
        # Load your parallel corpus
        self.sources = self.load_file(source_file)
        self.targets = self.load_file(target_file)
    
    def __getitem__(self, idx):
        src = self.encode(self.sources[idx])
        tgt = self.encode(self.targets[idx])
        return src, tgt
```

## 모델 구조 비교

| 구성 요소 | Decoder-only | Encoder-Decoder |
|----------|--------------|-----------------|
| Encoder | ❌ 없음 | ✅ 있음 |
| Decoder | ✅ Causal | ✅ Causal |
| Cross-Attention | ❌ 없음 | ✅ 있음 |
| 입력 | 1개 | 2개 (source + target) |
| Mask | Causal | Encoder: None<br>Decoder: Causal |
| 위치 인코딩 | RoPE | Sinusoidal |
| 용도 | LM, 생성 | 번역, 요약, Seq2Seq |

## 주요 코드 구성

### Encoder Layer
```python
class EncoderLayer:
    def forward(x, mask):
        # 1. Self-Attention (bidirectional)
        x = x + self_attn(x, x, x, mask)
        
        # 2. Feed-Forward
        x = x + ff(x)
        
        return x
```

### Decoder Layer
```python
class DecoderLayer:
    def forward(x, encoder_out, self_mask, cross_mask):
        # 1. Self-Attention (causal)
        x = x + self_attn(x, x, x, self_mask)
        
        # 2. Cross-Attention (attend to encoder)
        x = x + cross_attn(x, encoder_out, encoder_out, cross_mask)
        
        # 3. Feed-Forward
        x = x + ff(x)
        
        return x
```

## 성능 팁

1. **학습 속도 향상**
   - Batch size 증가 (GPU 메모리 허용 시)
   - Mixed precision 활성화 (기본 켜짐)
   - Warmup steps 조정

2. **정확도 향상**
   - More epochs
   - Larger model (dim, depth 증가)
   - Learning rate 튜닝

3. **메모리 절약**
   - Batch size 감소
   - Model size 감소
   - Gradient checkpointing (추가 구현 필요)

## 문제 해결

### Out of Memory
```yaml
# configs/enc_dec_config.yaml
training:
  batch_size: 32  # 64 → 32로 감소
```

### 학습이 느림
```yaml
data:
  num_workers: 8  # 4 → 8로 증가
```

### 정확도가 낮음
```yaml
training:
  num_epochs: 100  # 50 → 100으로 증가
  learning_rate: 1.0e-4  # 3e-4 → 1e-4로 감소
```

## 다음 단계

1. **실제 데이터로 학습**
   - 번역 corpus (e.g., WMT)
   - 요약 dataset (e.g., CNN/DailyMail)

2. **모델 개선**
   - Beam search decoding
   - Label smoothing
   - Layer normalization 위치 변경

3. **평가 지표 추가**
   - BLEU score
   - ROUGE score
   - METEOR

## 참고 자료

- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [PyTorch Seq2Seq Tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

## 질문

- **Q: Decoder-only vs Encoder-Decoder 중 뭘 쓸까?**
  - A: 언어 모델/생성 → Decoder-only, 번역/요약/Seq2Seq → Encoder-Decoder

- **Q: 두 모델을 동시에 학습할 수 있나?**
  - A: 네, 독립적으로 실행 가능 (다른 checkpoint 폴더 사용)

- **Q: 기존 Decoder-only 모델은 어떻게 되나?**
  - A: 그대로 유지됩니다. 두 모델 모두 사용 가능합니다.

