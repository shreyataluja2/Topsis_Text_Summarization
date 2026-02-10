# GPU-Accelerated TOPSIS Text Summarization Evaluation

**Fast, GPU-optimized model selection using TOPSIS for text summarization**

## Overview

This project evaluates pretrained text summarization models (T5, Flan-T5) using GPU acceleration and ranks them with **TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)** multi-criteria decision analysis.

### Key Features

- ✅ **GPU-Optimized**: FP16 inference for maximum speed
- ✅ **TOPSIS Ranking**: Multi-criteria decision framework
- ✅ **Production-Ready**: Clean, modular Python code
- ✅ **Colab Compatible**: Runs seamlessly on Google Colab
- ✅ **Comprehensive Metrics**: ROUGE-L, BLEU, inference time

---

## Quick Start

### Option 1: Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/angad2803/Data-Science-Assgns/blob/main/gpu_topsis_summarization/topsis_evaluation.ipynb)

**Just click and run!** Colab provides free GPU access.

### Option 2: Local Execution (GPU Required)

```bash
# Clone or download project
cd gpu_topsis_summarization

# Install dependencies
pip install -r requirements.txt

# Run evaluation
python topsis_gpu_evaluation.py
```

**Requirements:** CUDA-compatible GPU, Python 3.8+

---

## Methodology

### Models Evaluated

- google/flan-t5-base (250M parameters)
- t5-small (60M parameters)  
- google/flan-t5-small (80M parameters)

### Evaluation Metrics

- ROUGE-L (quality)
- BLEU (quality)
- Inference Time (efficiency)

### TOPSIS Framework

Weights used:
- ROUGE-L: 40%
- BLEU: 40%
- Time: 20%

Criteria: ROUGE-L (benefit), BLEU (benefit), Time (cost)

---

## Actual Experimental Results

### Final Rankings

```
======================================================================
FINAL RANKINGS
======================================================================
 Rank           Model  ROUGE-L     BLEU     Time  TOPSIS Score
    1 Flan-T5-Small   0.307895 0.066848 0.484830      0.808069 🥇
    2  Flan-T5-Base   0.407895 0.059283 3.819942      0.580347 🥈
    3      T5-Small   0.215385 0.012271 0.376374      0.389464 🥉
======================================================================

🏆 RECOMMENDED MODEL: Flan-T5-Small
   TOPSIS Score: 0.808069
   ROUGE-L: 0.3079
   BLEU: 0.0668
   Avg Time: 0.485s
```

### Results Table

| Rank | Model         | ROUGE-L  | BLEU     | Time (s) | TOPSIS Score |
| ---- | ------------- | -------- | -------- | -------- | ------------ |
| 🥇 1 | Flan-T5-Small | 0.307895 | 0.066848 | 0.484830 | **0.808069** |
| 🥈 2 | Flan-T5-Base  | 0.407895 | 0.059283 | 3.819942 | 0.580347     |
| 🥉 3 | T5-Small      | 0.215385 | 0.012271 | 0.376374 | 0.389464     |

### Key Insights

**Why Flan-T5-Small Won Despite Lower Quality Scores:**

The counterintuitive result where Flan-T5-Small ranked #1 despite having middle-tier ROUGE-L (0.308 vs 0.408 for Flan-T5-Base) demonstrates TOPSIS's strength in balancing multiple criteria:

1. **Speed Advantage (7.9x faster)**: Flan-T5-Small completed inference in 0.485s vs 3.82s for Flan-T5-Base — a massive **687% speedup**

2. **Time Weight Impact (20%)**: While quality metrics (ROUGE-L + BLEU) hold 80% combined weight, the extreme speed difference created strong TOPSIS performance

3. **Multi-Criteria Trade-off**: The 32% lower ROUGE-L score was offset by:
   - 7.9x faster inference (critical for production)
   - Better BLEU score (0.067 vs 0.059)
   - Lower computational cost

4. **TOPSIS Mathematics**: The Euclidean distance to the ideal solution heavily penalized Flan-T5-Base's 3.82s inference time (furthest from ideal on time axis), while Flan-T5-Small's balanced performance across all dimensions yielded the highest relative closeness score (0.808)

**Practical Implication:** For real-time applications or high-throughput scenarios, Flan-T5-Small offers the best quality-per-second ratio, making it the optimal choice despite not having the highest absolute quality.

---

## Results Visualization

![TOPSIS Rankings Chart](download.png)

**Chart Analysis:**

The bar chart visualizes the final TOPSIS scores with medal indicators:

- **Green Bar (Flan-T5-Small)**: Tallest bar at 0.808 - winner due to exceptional speed-quality balance
- **Blue Bar (Flan-T5-Base)**: Height 0.580 - penalized by 7.9x slower inference despite best quality
- **Yellow Bar (T5-Small)**: Height 0.389 - lowest due to poor quality metrics (BLEU=0.012)

The visualization clearly shows that **speed efficiency** combined with acceptable quality creates the highest TOPSIS score, validating the multi-criteria decision framework.

---

## Project Structure

```
gpu_topsis_summarization/
│
├── topsis_gpu_evaluation.py    # Main Python script
├── topsis_evaluation.ipynb     # Jupyter/Colab notebook
├── requirements.txt            # Dependencies
├── README.md                   # This file
│
├── topsis_results.csv          # Generated results
└── results_chart.png           # Generated visualization
```

---

## Customization

### Modify Evaluation Dataset

Edit the `dataset` list in the script (line 31):

```python
dataset = [
    {
        "text": "Your custom text here...",
        "summary": "Your reference summary..."
    },
    # Add more samples
]
```

### Change TOPSIS Weights

Modify weights in `main()` function (line 258):

```python
# Example: Prioritize speed over quality
weights = np.array([0.3, 0.3, 0.4])  # ROUGE, BLEU, Time
```

### Add New Models

Update `models_to_test` dictionary (line 51):

```python
models_to_test = {
    "Flan-T5-Base": "google/flan-t5-base",
    "Your-Model": "huggingface/your-model-name"
}
```

---

**GitHub:**
<https://github.com/shreyataluja2/Topsis_Text_Summarization.git>
