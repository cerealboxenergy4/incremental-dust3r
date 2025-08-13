# DUSt3R 점진적 최적화 (Incremental Optimization)

## 개요

DUSt3R의 점진적 최적화 기능은 기존에 최적화된 글로벌 씬에 새로운 이미지를 하나씩 추가하면서 최적화를 수행할 수 있도록 해줍니다. 이는 대규모 씬 재구성에서 효율적인 처리를 가능하게 합니다.

## 주요 특징

- **기존 최적화 상태 유지**: 이미 최적화된 이미지들의 파라미터를 보존
- **점진적 이미지 추가**: 새로운 이미지를 하나씩 추가 가능
- **선택적 파라미터 고정**: 기존 이미지의 파라미터를 고정하고 새로운 이미지만 최적화 가능
- **효율적인 메모리 사용**: 전체 씬을 다시 계산하지 않고 증분적으로 처리

## 사용법

### 1. 기본 사용법

```python
from dust3r.cloud_opt.optimizer import PointCloudOptimizer
from dust3r.cloud_opt.incremental_optimizer import IncrementalPCOptimizer

# 1. 기존 씬 최적화
base_optimizer = PointCloudOptimizer(view1, view2, pred1, pred2)
base_loss = base_optimizer.compute_global_alignment(init='msp', niter=300)

# 2. 새로운 이미지 추가
incremental_optimizer = IncrementalPCOptimizer(
    base_optimizer,
    new_view,
    new_pred1,
    new_pred2
)

# 3. 점진적 최적화 (기존 파라미터 고정)
incremental_loss = incremental_optimizer.optimize_incremental(
    lr=0.01,
    niter=100,
    freeze_existing=True
)
```

### 2. 여러 이미지 순차 추가

```python
# 첫 번째 이미지 추가
optimizer1 = IncrementalPCOptimizer(base_optimizer, view1, pred1, pred2)
loss1 = optimizer1.optimize_incremental(freeze_existing=True)

# 두 번째 이미지 추가
optimizer2 = optimizer1.add_new_image(view2, pred1_2, pred2_2)
loss2 = optimizer2.optimize_incremental(freeze_existing=True)

# 세 번째 이미지 추가
optimizer3 = optimizer2.add_new_image(view3, pred1_3, pred2_3)
loss3 = optimizer3.optimize_incremental(freeze_existing=True)
```

### 3. 최적화 상태 확인

```python
# 최적화 요약 정보 확인
summary = incremental_optimizer.get_optimization_summary()
print(f"총 이미지 수: {summary['total_images']}")
print(f"기존 이미지 수: {summary['existing_images']}")
print(f"새로운 이미지 수: {summary['new_images']}")
print(f"훈련 가능한 파라미터 수: {summary['trainable_parameters']}")

# 새로운 이미지 인덱스 확인
new_indices = incremental_optimizer.get_new_image_indices()
print(f"새로 추가된 이미지 인덱스: {new_indices}")
```

## 클래스 구조

### IncrementalPCOptimizer

`BasePCOptimizer`를 상속받아 점진적 최적화 기능을 제공합니다.

#### 주요 메서드

- `__init__(base_optimizer, new_view, new_pred1, new_pred2, **kwargs)`
  - 기존 최적화된 씬과 새로운 이미지 데이터로 초기화

- `add_new_image(new_view, new_pred1, new_pred2, **kwargs)`
  - 현재 씬에 새로운 이미지 추가

- `optimize_incremental(lr=0.01, niter=100, schedule='cosine', lr_min=1e-6, freeze_existing=True, **kwargs)`
  - 점진적 최적화 수행

- `get_new_image_indices()`
  - 새로 추가된 이미지의 인덱스 반환

- `get_optimization_summary()`
  - 최적화 상태 요약 정보 반환

## 최적화 전략

### 1. 파라미터 고정 전략

- `freeze_existing=True`: 기존 이미지의 파라미터를 고정하고 새로운 이미지만 최적화
- `freeze_existing=False`: 모든 파라미터를 함께 최적화

### 2. 학습률 조정

```python
# 새로운 이미지에 대해서는 더 높은 학습률 사용
incremental_loss = incremental_optimizer.optimize_incremental(
    lr=0.02,  # 새로운 이미지에 대해 더 높은 학습률
    niter=150,
    freeze_existing=True
)
```

### 3. 반복 횟수 조정

```python
# 첫 번째 추가: 많은 반복
loss1 = optimizer1.optimize_incremental(niter=200, freeze_existing=True)

# 후속 추가: 적은 반복
loss2 = optimizer2.optimize_incremental(niter=100, freeze_existing=True)
```

## 성능 고려사항

### 1. 메모리 효율성

- 기존 파라미터를 고정하면 메모리 사용량 감소
- 그래디언트 계산이 새로운 파라미터에만 집중

### 2. 계산 효율성

- 기존 씬의 최적화 상태를 재사용
- 새로운 이미지에 대해서만 추가 계산 필요

### 3. 수렴성

- 기존 파라미터가 고정되어 있어 수렴이 더 안정적
- 새로운 이미지의 초기화가 중요

## 예제 실행

```bash
# 예제 스크립트 실행
python dust3r/examples/incremental_optimization_example.py
```

## 주의사항

1. **데이터 일관성**: 새로운 이미지의 예측 데이터가 기존 이미지와 일치해야 함
2. **인덱스 관리**: 이미지 인덱스가 올바르게 설정되어야 함
3. **메모리 관리**: 대규모 씬에서는 메모리 사용량을 모니터링해야 함
4. **초기화**: 새로운 이미지의 초기 파라미터 설정이 중요

## 확장 가능성

- **배치 처리**: 여러 이미지를 동시에 추가하는 기능
- **적응적 파라미터 고정**: 중요도에 따른 선택적 파라미터 고정
- **메타러닝**: 이전 최적화 경험을 활용한 효율적인 초기화

