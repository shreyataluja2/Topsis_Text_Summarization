import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import torch
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

def setup_nltk():
    try:
        nltk.download('punkt', quiet=True)
        print("✓ NLTK data downloaded")
    except Exception as e:
        print(f"Warning: NLTK download failed - {e}")

dataset = [
    {
        "text": "Artificial intelligence is transforming industries worldwide. From healthcare to finance, AI systems are improving efficiency, decision-making, and automation. However, ethical concerns remain regarding bias, transparency, and employment impact.",
        "summary": "AI is transforming industries but raises ethical concerns."
    },
    {
        "text": "Climate change is one of the most pressing global challenges. Rising temperatures and extreme weather events threaten ecosystems and human life. Governments are investing in renewable energy to reduce carbon emissions.",
        "summary": "Climate change threatens the planet, pushing renewable energy efforts."
    },
    {
        "text": "Remote work has become the new normal after the pandemic. Companies are adopting hybrid models that combine office and home work. This shift requires new management approaches and collaboration tools.",
        "summary": "Remote work is now standard, requiring new tools and approaches."
    },
    {
        "text": "Electric vehicles are gaining market share as battery technology improves and charging infrastructure expands. Major automakers are committing to electric-only lineups by 2030. Government incentives are accelerating adoption.",
        "summary": "Electric vehicles are growing due to better tech and infrastructure."
    }
]

models_to_test = {
    "Flan-T5-Base": "google/flan-t5-base",
    "T5-Small": "t5-small",
    "Flan-T5-Small": "google/flan-t5-small"
}

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
smoothie = SmoothingFunction().method1

def evaluate_on_gpu(name, path):
    print(f"\n--- Benchmarking {name} on GPU ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        print("⚠ Warning: GPU not available, falling back to CPU")
    else:
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        path, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="eager"
    ).to(device)
    model.eval()

    rouge_scores, bleu_scores, times = [], [], []

    for i, item in enumerate(dataset, 1):
        input_text = "summarize: " + item["text"]
        
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=512
        ).to(device)
        
        start = time.time()
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"], 
                max_length=60, 
                min_length=10, 
                num_beams=4,
                early_stopping=True
            )
        end = time.time()
        
        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        rouge = scorer.score(item["summary"], output)["rougeL"].fmeasure
        bleu = sentence_bleu(
            [item["summary"].split()], 
            output.split(), 
            smoothing_function=smoothie
        )

        rouge_scores.append(rouge)
        bleu_scores.append(bleu)
        times.append(end - start)
        
        print(f"  Sample {i}/{len(dataset)}: ROUGE-L={rouge:.4f}, BLEU={bleu:.4f}, Time={end-start:.4f}s")
        
    del model, tokenizer, inputs, summary_ids
    if device == "cuda":
        torch.cuda.empty_cache()

    avg_rouge = np.mean(rouge_scores)
    avg_bleu = np.mean(bleu_scores)
    avg_time = np.mean(times)
    
    print(f"✓ Success: {name} | Avg Time: {avg_time:.4f}s | ROUGE-L: {avg_rouge:.4f} | BLEU: {avg_bleu:.4f}")
    
    return avg_rouge, avg_bleu, avg_time

def apply_topsis(df, weights, criteria):
    print("\n" + "="*70)
    print("APPLYING TOPSIS MULTI-CRITERIA DECISION ANALYSIS")
    print("="*70)
    
    data = df[["ROUGE-L", "BLEU", "Time"]].values.astype(float)
    
    print(f"\nWeights: ROUGE-L={weights[0]}, BLEU={weights[1]}, Time={weights[2]}")
    print(f"Criteria: ROUGE-L=Benefit, BLEU=Benefit, Time=Cost")
    
    norm = data / (np.sqrt((data**2).sum(axis=0)) + 1e-9)
    weighted = norm * weights
    
    ideal_best = np.where(criteria == 1, weighted.max(0), weighted.min(0))
    ideal_worst = np.where(criteria == 1, weighted.min(0), weighted.max(0))
    
    print(f"\nIdeal Best:  {ideal_best}")
    print(f"Ideal Worst: {ideal_worst}")
    
    distance_best = np.sqrt(((weighted - ideal_best)**2).sum(axis=1))
    distance_worst = np.sqrt(((weighted - ideal_worst)**2).sum(axis=1))
    
    df["TOPSIS Score"] = distance_worst / (distance_best + distance_worst + 1e-9)
    
    df = df.sort_values("TOPSIS Score", ascending=False)
    df["Rank"] = range(1, len(df) + 1)
    
    return df

def create_ranking_chart(df, save_path="results_chart.png"):
    plt.figure(figsize=(10, 6))
    
    colors = ['#4CAF50', '#2196F3', '#FFC107'][:len(df)]
    bars = plt.bar(df["Model"], df["TOPSIS Score"], color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, score in zip(bars, df["TOPSIS Score"]):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            height + 0.02, 
            f'{score:.4f}',
            ha='center', 
            va='bottom',
            fontweight='bold',
            fontsize=10
        )
    
    medals = {1: '🥇', 2: '🥈', 3: '🥉'}
    for i, (idx, row) in enumerate(df.iterrows()):
        if row['Rank'] in medals:
            plt.text(
                i, 
                0.02, 
                medals[row['Rank']], 
                ha='center', 
                fontsize=20
            )
    
    plt.title("GPU-Accelerated NLP Model Rankings (TOPSIS)", fontsize=14, fontweight='bold', pad=20)
    plt.ylabel("TOPSIS Performance Score", fontsize=12, fontweight='bold')
    plt.xlabel("Model", fontsize=12, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Chart saved to: {save_path}")
    
    plt.show()

def main():
    print("="*70)
    print("GPU-ACCELERATED TOPSIS MODEL EVALUATION FOR TEXT SUMMARIZATION")
    print("="*70)
    
    setup_nltk()
    
    if torch.cuda.is_available():
        print(f"\n✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\n⚠ GPU not available - using CPU (slower)")
    
    print("\n" + "="*70)
    print("STEP 1: MODEL EVALUATION")
    print("="*70)
    
    results = []
    for name, path in models_to_test.items():
        try:
            r, b, t = evaluate_on_gpu(name, path)
            results.append([name, r, b, t])
        except Exception as e:
            print(f"\n❌ Failed {name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    if len(results) >= 2:
        print("\n" + "="*70)
        print("STEP 2: TOPSIS RANKING")
        print("="*70)
        
        df = pd.DataFrame(results, columns=["Model", "ROUGE-L", "BLEU", "Time"])
        
        weights = np.array([0.4, 0.4, 0.2])
        criteria = np.array([1, 1, 0])
        
        df = apply_topsis(df, weights, criteria)
        
        print("\n" + "="*70)
        print("FINAL RANKINGS")
        print("="*70)
        print(df[["Rank", "Model", "ROUGE-L", "BLEU", "Time", "TOPSIS Score"]].to_string(index=False))
        print("="*70)
        
        winner = df.iloc[0]
        print(f"\n🏆 RECOMMENDED MODEL: {winner['Model']}")
        print(f"   TOPSIS Score: {winner['TOPSIS Score']:.4f}")
        print(f"   ROUGE-L: {winner['ROUGE-L']:.4f}")
        print(f"   BLEU: {winner['BLEU']:.4f}")
        print(f"   Avg Time: {winner['Time']:.4f}s")
        
        df.to_csv("topsis_results.csv", index=False)
        print(f"\n✓ Results saved to: topsis_results.csv")
        
        create_ranking_chart(df)
        
    else:
        print("\n❌ Error: Not enough successful models to rank.")
        print(f"   Only {len(results)} model(s) evaluated successfully.")
        print("   Need at least 2 models for TOPSIS comparison.")

if __name__ == "__main__":
    main()
