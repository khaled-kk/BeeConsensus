# 🐝 BeeConsensus Framework

**BeeConsensus** is a swarm-intelligence framework designed to mitigate LLM hallucinations through multi-persona deliberation, semantic clustering, and consensus-based truth filtering.

Inspired by the collective decision-making of honeybees, this system orchestrates multiple "Scout" agents to explore different perspectives on a query before reaching a unified, factual consensus.

---

## ✨ Key Features
- **Swarm Deliberation**: Multiple specialized agents (Factual Specialist, Cautious Reasoner, Devil's Advocate) deliberate on every query.
- **Semantic Clustering**: Uses `DBSCAN` and `Sentence-Transformers` to group similar answers and filter out "noisy" hallucinations.
- **Automatic Calibration**: Every answer is assigned a confidence score based on cluster cohesion and dominance.
- **Cross-Platform Support**: Optimized for **OpenVINO** (Intel hardware) and **CUDA** (NVIDIA GPUs/Kaggle/Colab).
- **Persistent Checkpoints**: Automatically saves progress to CSV, allowing for seamless resumption of long-running benchmarks.

---

## 🚀 Execution Guide

### 1. Local Setup
```bash
# Clone the repository
git clone https://github.com/khaled-kk/BeeConsensus.git
cd BeeConsensus

# Install dependencies
pip install -r requirements.txt

# Run a quick demo
python evaluate.py --mode demo
```

### 2. Kaggle Deployment (Recommended for Benchmarking)
For running the full **TruthfulQA** dataset, Kaggle's dual T4 GPUs are highly effective.

**Environment Setup:**
```python
# 1. Install required libraries
!pip install -q transformers accelerate bitsandbytes sentence-transformers datasets

# 2. Login to Hugging Face (Required for Llama-3.1 access)
from huggingface_hub import login
login("YOUR_HF_TOKEN")

# 3. Import your script files (if using Kaggle datasets)
!find /kaggle/input -name "*.py" -exec cp {} . \;
```

**Run Benchmark:**
```bash
python evaluate.py --mode truthfulqa --model_id "meta-llama/Llama-3.1-8B-Instruct" --limit 817
```

---

## 🔄 Resuming Progress
BeeConsensus is built for reliability. If your cloud session times out or you need to pause:
- The script automatically saves results to `benchmark_progress.csv` after **every question**.
- When you restart the script, it will detect the existing CSV, reconstruct the current accuracy stats, and **automatically skip** already answered questions.
- Results are appended to the file, ensuring you never lose data.

---

## 📊 Methodology
BeeConsensus operates in three distinct phases:
1. **Scouting**: Diverse agents generate independent candidate answers using varied system prompts.
2. **Clustering**: Semantic embeddings group these answers into "belief clusters" using cosine similarity distance.
3. **Consensus (Quorum)**: A dominant cluster must meet a quorum threshold (e.g., 60-80%) to be accepted. If no consensus is reached, agents re-deliberate with the context of their peers' answers.

---

*Developed by [Khaled Walid](https://github.com/khaled-kk)*
