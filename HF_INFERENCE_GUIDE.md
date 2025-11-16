# Hugging Face Encoder-Decoder Inference Guide

실전에서 바로 사용 가능한 Hugging Face 사전학습 Encoder-Decoder 모델 inference 가이드입니다.

## 프로젝트 구성

```
transformer-ddp-lm/
├── hf_inference_single_gpu.py       # 단일 GPU inference
├── hf_inference_multi_gpu.py        # 멀티 GPU inference (DDP)
├── notebooks/
│   └── databricks_hf_inference.ipynb # Databricks 노트북
└── HF_INFERENCE_GUIDE.md            # 이 파일
```

## 사용 가능한 모델

### 1. BART (Facebook)

```python
# Summarization에 최적화
"facebook/bart-base"              # 140M params
"facebook/bart-large"             # 400M params  
"facebook/bart-large-cnn"         # CNN/DailyMail 파인튠
"facebook/bart-large-xsum"        # XSum 파인튠
```

### 2. T5 (Google)

```python
# 범용 Seq2Seq
"t5-small"                        # 60M params
"t5-base"                         # 220M params
"t5-large"                        # 770M params
"t5-3b"                           # 3B params (GPU 메모리 주의!)
```

### 3. Pegasus (Google)

```python
# Summarization 전문
"google/pegasus-xsum"
"google/pegasus-cnn_dailymail"
"google/pegasus-multi_news"
```

### 4. mT5 (Multilingual)

```python
# 다국어 지원
"google/mt5-small"
"google/mt5-base"
"google/mt5-large"
```

## 빠른 시작

### 단일 GPU

```bash
python hf_inference_single_gpu.py \
    --model-name facebook/bart-large-cnn \
    --dataset-name cnn_dailymail \
    --num-samples 100 \
    --batch-size 8
```

### 멀티 GPU (8개)

```bash
torchrun --nproc_per_node=8 hf_inference_multi_gpu.py \
    --model-name facebook/bart-large-cnn \
    --dataset-name cnn_dailymail \
    --num-samples 1000 \
    --batch-size 4
```

### Databricks

노트북 실행:
```python
# notebooks/databricks_hf_inference.ipynb
```

## 데이터셋

### CNN/DailyMail (Summarization)

```python
--dataset-name cnn_dailymail
```

- **Task**: 뉴스 기사 요약
- **Size**: 300K+ articles
- **Source**: 뉴스 기사 본문
- **Target**: 요약문 (highlights)

### XSum (Extreme Summarization)

```python
--dataset-name xsum
```

- **Task**: 극단적으로 짧은 요약
- **Size**: 200K+ documents
- **Source**: BBC 기사
- **Target**: 한 문장 요약

## 상세 사용법

### 모델 선택

```python
# Summarization
--model-name facebook/bart-large-cnn        # 추천!

# Multilingual
--model-name google/mt5-base

# Large model
--model-name t5-large
```

### 파라미터 조정

```bash
--batch-size 4              # GPU 메모리에 따라 조정
--max-length 128            # 생성할 최대 길이
--num-beams 4               # Beam search 빔 개수
--num-samples 1000          # 테스트할 샘플 수
```

### GPU 메모리 가이드

| Model | Size | Batch Size (16GB GPU) | Batch Size (24GB GPU) |
|-------|------|----------------------|----------------------|
| BART-base | 140M | 16 | 32 |
| BART-large | 400M | 4 | 8 |
| T5-base | 220M | 8 | 16 |
| T5-large | 770M | 2 | 4 |

## 성능 비교

### Single GPU vs Multi-GPU (8 GPUs)

| Setup | Model | Samples | Time | Speed |
|-------|-------|---------|------|-------|
| 1x A10 | BART-large-cnn | 1000 | ~15 min | 1.1 samples/sec |
| 8x A10 | BART-large-cnn | 1000 | ~2 min | 8.3 samples/sec |

**속도 향상: ~7.5x** (거의 linear scaling!)

## 평가 지표

### ROUGE Score

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
score = scorer.score(reference, generated)
```

- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap  
- **ROUGE-L**: Longest common subsequence

### Expected Scores (BART-large-cnn on CNN/DailyMail)

```
ROUGE-1: ~44%
ROUGE-2: ~21%
ROUGE-L: ~41%
```

## 실전 예시

### 예시 1: 뉴스 기사 요약

```python
# Source (truncated)
"(CNN) -- Usher is engaged to his business partner and girlfriend, Grace Miguel. 
The R&B singer confirmed the news through a statement to \"Us Weekly.\" 
\"I have an incredible partner and manager. She has helped me through some of 
the most challenging times...\" [continues...]

# Generated Summary
Usher is engaged to his girlfriend and business partner Grace Miguel. 
The R&B singer confirmed the news through a statement to Us Weekly.

# Ground Truth
Usher confirms he's engaged to his girlfriend Grace Miguel.
The couple have been dating for several years.
```

### 예시 2: XSum

```python
# Source
A police force has been criticised after a tweet about a missing person was sent 
more than 12 hours after she disappeared. [continues...]

# Generated
Police criticised for delayed missing person tweet.

# Ground Truth  
Police criticised over delayed tweet about missing woman.
```

## 커스텀 데이터 사용

### 자신의 데이터로 inference

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).cuda()

# Your custom text
texts = [
    "Your article text here...",
    "Another article..."
]

inputs = tokenizer(texts, max_length=1024, truncation=True, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=128, num_beams=4)
summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)

for summary in summaries:
    print(summary)
```

## Databricks 실행

### 클러스터 설정

```
Runtime: DBR 13.3 ML+
Driver: 8x A10 (g5.48xlarge) or 8x V100 (p3.16xlarge)
Workers: 0
```

### 노트북 실행

1. `notebooks/databricks_hf_inference.ipynb` import
2. 클러스터 attach
3. 셀 순서대로 실행

## 고급 사용

### 커스텀 생성 설정

```python
outputs = model.generate(
    **inputs,
    max_length=128,
    min_length=30,              # 최소 길이
    num_beams=4,                # Beam search
    length_penalty=2.0,         # 길이 패널티
    early_stopping=True,        # 조기 종료
    no_repeat_ngram_size=3,     # N-gram 반복 방지
    temperature=1.0,            # Sampling temperature
    top_k=50,                   # Top-k sampling
    top_p=0.95                  # Nucleus sampling
)
```

### 배치 처리 최적화

```python
# 동적 패딩으로 메모리 절약
from transformers import DataCollatorForSeq2Seq

collator = DataCollatorForSeq2Seq(tokenizer, model=model)
```

## 문제 해결

### OOM (Out of Memory)

```python
# 해결책
--batch-size 2              # 배치 사이즈 감소
--max-length 64             # 생성 길이 감소
--num-beams 1               # Greedy decoding 사용
```

### 느린 inference

```python
# 해결책
torch_dtype=torch.float16   # FP16 사용
device_map="auto"           # 자동 디바이스 배치
```

### CUDA OOM with Multi-GPU

```bash
# GPU당 배치 사이즈 감소
--batch-size 2

# 또는 GPU 개수 감소
torchrun --nproc_per_node=4 ...  # 8 → 4
```

## From Scratch vs Hugging Face 비교

| 항목 | From Scratch | Hugging Face |
|------|--------------|--------------|
| **개발 시간** | 몇 주 | 몇 분 |
| **성능** | 낮음 (작은 데이터) | 높음 (사전학습) |
| **학습 필요** | ✓ (필수) | ✗ (optional) |
| **데이터 필요** | 많음 | 적음/없음 |
| **실용성** | 학습용 | 실전용 ⭐ |
| **비용** | 높음 | 낮음 |

**결론**: 실전에서는 Hugging Face 사용!

## 다음 단계

1. **Fine-tuning**: 자신의 데이터로 fine-tune
2. **평가**: ROUGE, BLEU 등으로 정량 평가
3. **최적화**: TorchScript, ONNX로 속도 개선
4. **배포**: FastAPI, Triton으로 서빙

## 참고 자료

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [BART Paper](https://arxiv.org/abs/1910.13461)
- [T5 Paper](https://arxiv.org/abs/1910.10683)
- [Pegasus Paper](https://arxiv.org/abs/1912.08777)

