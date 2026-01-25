"""
Multi-News 데이터셋 구성 시각화
데이터셋: Awesome075/multi_news_parquet (Parquet 버전 - 최신 datasets 호환)
원본: alexfabbri/multi_news
"""

import os
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정 (서버 환경에서는 영어로 대체)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("데이터셋 로딩 중...")
# Parquet 버전 사용 (최신 datasets 라이브러리 호환)
dataset = load_dataset("Awesome075/multi_news_parquet")
print("데이터셋 로딩 완료!")

# 데이터셋 기본 정보
print("\n=== 데이터셋 구조 ===")
print(dataset)
print("\n=== 컬럼 정보 ===")
print(dataset['train'].features)

# 각 split의 샘플 수
splits = list(dataset.keys())
split_sizes = [len(dataset[split]) for split in splits]

# 텍스트 길이 분석 (샘플링)
sample_size = min(1000, len(dataset['train']))
train_sample = dataset['train'].select(range(sample_size))

# document 길이 (단어 수)
doc_lengths = [len(item['document'].split()) for item in train_sample]

# summary 길이 (단어 수)
summary_lengths = [len(item['summary'].split()) for item in train_sample]

# 소스 문서 개수 (|||||로 구분됨)
num_sources = [item['document'].count('|||||') + 1 for item in train_sample]

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Multi-News Dataset Composition Analysis', fontsize=16, fontweight='bold')

# 1. Split 별 샘플 수
ax1 = axes[0, 0]
colors = ['#3498db', '#2ecc71', '#e74c3c']
bars = ax1.bar(splits, split_sizes, color=colors[:len(splits)], edgecolor='black', linewidth=1.2)
ax1.set_title('Number of Samples per Split', fontsize=12, fontweight='bold')
ax1.set_xlabel('Split')
ax1.set_ylabel('Number of Samples')
for bar, size in zip(bars, split_sizes):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
             f'{size:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.set_ylim(0, max(split_sizes) * 1.15)

# 2. Document 길이 분포
ax2 = axes[0, 1]
ax2.hist(doc_lengths, bins=50, color='#9b59b6', edgecolor='black', alpha=0.7)
ax2.axvline(np.mean(doc_lengths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(doc_lengths):.0f}')
ax2.axvline(np.median(doc_lengths), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(doc_lengths):.0f}')
ax2.set_title('Document Length Distribution (words)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Number of Words')
ax2.set_ylabel('Frequency')
ax2.legend()

# 3. Summary 길이 분포
ax3 = axes[1, 0]
ax3.hist(summary_lengths, bins=50, color='#1abc9c', edgecolor='black', alpha=0.7)
ax3.axvline(np.mean(summary_lengths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(summary_lengths):.0f}')
ax3.axvline(np.median(summary_lengths), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(summary_lengths):.0f}')
ax3.set_title('Summary Length Distribution (words)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Number of Words')
ax3.set_ylabel('Frequency')
ax3.legend()

# 4. 소스 문서 개수 분포
ax4 = axes[1, 1]
unique_sources, counts = np.unique(num_sources, return_counts=True)
ax4.bar(unique_sources, counts, color='#f39c12', edgecolor='black', alpha=0.8)
ax4.set_title('Number of Source Documents Distribution', fontsize=12, fontweight='bold')
ax4.set_xlabel('Number of Source Documents')
ax4.set_ylabel('Frequency')
ax4.set_xticks(unique_sources[::2] if len(unique_sources) > 10 else unique_sources)

# 통계 정보 텍스트 박스
stats_text = f"""Dataset Statistics (sample n={sample_size}):
- Document: mean={np.mean(doc_lengths):.0f}, std={np.std(doc_lengths):.0f} words
- Summary: mean={np.mean(summary_lengths):.0f}, std={np.std(summary_lengths):.0f} words
- Source docs: mean={np.mean(num_sources):.1f}, range=[{min(num_sources)}, {max(num_sources)}]"""

fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0.08, 1, 0.96])

# PNG로 저장 (스크립트와 동일한 디렉토리에 저장)
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, 'multi_news_visualization.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n시각화 저장 완료: {output_path}")

# 추가 통계 출력
print("\n=== 상세 통계 ===")
for split, size in zip(splits, split_sizes):
    print(f"{split.capitalize()} 샘플 수: {size:,}")
print(f"\nDocument 길이 (단어): mean={np.mean(doc_lengths):.1f}, median={np.median(doc_lengths):.1f}, std={np.std(doc_lengths):.1f}")
print(f"Summary 길이 (단어): mean={np.mean(summary_lengths):.1f}, median={np.median(summary_lengths):.1f}, std={np.std(summary_lengths):.1f}")
print(f"소스 문서 개수: mean={np.mean(num_sources):.2f}, min={min(num_sources)}, max={max(num_sources)}")
