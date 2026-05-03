# 🐝 BeeConsensus Framework

**BeeConsensus** is a swarm-intelligence framework designed to mitigate LLM hallucinations through multi-persona deliberation, semantic clustering, and consensus-based truth filtering.

Inspired by the collective decision-making of honeybees, this system orchestrates multiple "Scout" agents to explore different perspectives on a query before reaching a unified, factual consensus.

## ✨ Key Features
- **Swarm Deliberation**: Multiple specialized agents (Fact-Checkers, Critics, Synthesizers) deliberate on every query.
- **Semantic Clustering**: Uses `DBSCAN` and `Sentence-Transformers` to group similar answers and identify the true consensus.
- **Hallucination Mitigation**: Significantly improves TruthfulQA scores by forcing agents to provide evidence and cross-examine each other.
- **Cross-Platform Support**: Runs on **OpenVINO** (local Intel CPUs/GPUs) and **Native CUDA** (NVIDIA GPUs/Google Colab).

## 🚀 Quick Start (Local)

1. **Clone & Install**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/BeeConsensus.git
   cd BeeConsensus
   pip install -r requirements.txt
   ```

2. **Run the Benchmark**:
   ```bash
   python evaluate.py --mode truthfulqa --limit 10
   ```

## ☁️ Running in the Cloud (Colab & Kaggle)
BeeConsensus is optimized for high-performance cloud GPUs (NVIDIA T4, L4, A100). 

- **Google Colab**: Connect your Google Drive for persistent checkpoints.
- **Kaggle**: Use the 30-hours/week free GPU quota for long-running benchmarks.

### Execution:
1. Upload `beeconsensus.py` and `evaluate.py`.
2. Run the environment setup:
   ```bash
   pip install transformers bitsandbytes accelerate sentence-transformers
   ```
3. Run the full benchmark:
   ```bash
   python evaluate.py --model_id "meta-llama/Llama-3.1-8B-Instruct" --limit 0
   ```

## 📊 Methodology
BeeConsensus operates in three distinct phases:
1. **Scouting**: Diverse agents generate independent candidate answers.
2. **Clustering**: Semantic embeddings group these answers into "belief clusters."
3. **Consensus**: The framework selects the most dominant and cohesive cluster to formulate the final truthful response.

---
*Created with 🐝 by [Khaled Walid]*
