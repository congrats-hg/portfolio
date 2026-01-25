import matplotlib.pyplot as plt
from datasets import load_dataset

# 데이터셋 로드
dataset = load_dataset("HuggingFaceM4/ChartQA")
train_dataset = dataset["train"]
val_dataset = dataset["val"]
test_dataset = dataset["test"]

# 데이터 수집
train_widths, train_heights, train_query_lens = [], [], []
val_widths, val_heights, val_query_lens = [], [], []
test_widths, test_heights, test_query_lens = [], [], []

for sample in train_dataset:
    w, h = sample["image"].size
    train_widths.append(w)
    train_heights.append(h)
    train_query_lens.append(len(sample["query"]))

for sample in val_dataset:
    w, h = sample["image"].size
    val_widths.append(w)
    val_heights.append(h)
    val_query_lens.append(len(sample["query"]))

for sample in test_dataset:
    w, h = sample["image"].size
    test_widths.append(w)
    test_heights.append(h)
    test_query_lens.append(len(sample["query"]))

# 시각화
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 이미지 너비 분포
axes[0, 0].hist(train_widths, bins=50, alpha=0.7, label='Train')
axes[0, 0].hist(val_widths, bins=50, alpha=0.7, label='Val')
axes[0, 0].hist(test_widths, bins=50, alpha=0.7, label='Test')
axes[0, 0].set_title('Image Width Distribution')
axes[0, 0].legend()

# 이미지 높이 분포
axes[0, 1].hist(train_heights, bins=50, alpha=0.7, label='Train')
axes[0, 1].hist(val_heights, bins=50, alpha=0.7, label='Val')
axes[0, 1].hist(test_heights, bins=50, alpha=0.7, label='Test')
axes[0, 1].set_title('Image Height Distribution')
axes[0, 1].legend()

# 이미지 크기 (width x height) 산점도
axes[0, 2].scatter(train_widths, train_heights, alpha=0.3, s=5, label='Train')
axes[0, 2].scatter(val_widths, val_heights, alpha=0.3, s=5, label='Val')
axes[0, 2].scatter(test_widths, test_heights, alpha=0.3, s=5, label='Test')
axes[0, 2].set_xlabel('Width')
axes[0, 2].set_ylabel('Height')
axes[0, 2].set_title('Image Size (Width x Height)')
axes[0, 2].legend()

# Query 길이 분포
axes[1, 0].hist(train_query_lens, bins=50, alpha=0.7, label='Train')
axes[1, 0].hist(val_query_lens, bins=50, alpha=0.7, label='Val')
axes[1, 0].hist(test_query_lens, bins=50, alpha=0.7, label='Test')
axes[1, 0].set_title('Query Length Distribution')
axes[1, 0].legend()

# Query 길이 boxplot
axes[1, 1].boxplot([train_query_lens, val_query_lens, test_query_lens], tick_labels=['Train', 'Val', 'Test'])
axes[1, 1].set_title('Query Length Boxplot')

# 인덱스별 query 길이 (순서 패턴 확인용)
axes[1, 2].plot(train_query_lens[:500], alpha=0.7, label='Train (first 500)')
axes[1, 2].plot(val_query_lens[:500], alpha=0.7, label='Val (first 500)')
axes[1, 2].plot(test_query_lens[:500], alpha=0.7, label='Test (first 500)')
axes[1, 2].set_title('Query Length by Index (check ordering)')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('dataset_distribution.png', dpi=150)
print("Saved to dataset_distribution.png")
