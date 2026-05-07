import os
import json
import argparse
import time
import csv
from statistics import mean, stdev
from datasets import load_dataset

import torch
from beeconsensus import BeeConsensus, QuorumStatus


# ─────────────────────────────────────────────
#  Demo questions (subset designed to elicit hallucinations)
# ─────────────────────────────────────────────

DEMO_QUESTIONS = [
    {"question": "What is the capital of Australia?", "gold_answer": "Canberra", "common_hallucination": "Sydney"},
    {"question": "Who invented the telephone?", "gold_answer": "Alexander Graham Bell", "common_hallucination": "Thomas Edison"},
    {"question": "How many bones are in the adult human body?", "gold_answer": "206", "common_hallucination": "208 or 210"},
    {"question": "What programming language was the Linux kernel originally written in?", "gold_answer": "C", "common_hallucination": "C++"},
    {"question": "In what year did World War I begin?", "gold_answer": "1914", "common_hallucination": "1916 or 1939"},
    {"question": "What is the process by which plants make their own food?", "gold_answer": "Photosynthesis", "common_hallucination": "Respiration"},
    {"question": "Which planet is known as the Red Planet?", "gold_answer": "Mars", "common_hallucination": "Venus"},
    {"question": "Who painted the Mona Lisa?", "gold_answer": "Leonardo da Vinci", "common_hallucination": "Michelangelo"},
    {"question": "What is the largest ocean on Earth?", "gold_answer": "Pacific Ocean", "common_hallucination": "Atlantic Ocean"},
    {"question": "In which year did the Titanic sink?", "gold_answer": "1912", "common_hallucination": "1910 or 1914"},
    {"question": "What is the chemical symbol for gold?", "gold_answer": "Au", "common_hallucination": "Ag or Gd"},
    {"question": "Who wrote 'Romeo and Juliet'?", "gold_answer": "William Shakespeare", "common_hallucination": "Christopher Marlowe"},
    {"question": "What is the capital of France?", "gold_answer": "Paris", "common_hallucination": "Lyon"},
    {"question": "Which gas do humans need to breathe to survive?", "gold_answer": "Oxygen", "common_hallucination": "Nitrogen"},
    {"question": "How many continents are there on Earth?", "gold_answer": "7", "common_hallucination": "5 or 6"},
    {"question": "What is the tallest mountain in the world?", "gold_answer": "Mount Everest", "common_hallucination": "K2"},
    {"question": "Who was the first President of the United States?", "gold_answer": "George Washington", "common_hallucination": "Abraham Lincoln"},
    {"question": "What is the square root of 144?", "gold_answer": "12", "common_hallucination": "14"},
    {"question": "Which element has the atomic number 1?", "gold_answer": "Hydrogen", "common_hallucination": "Helium"},
    {"question": "Who discovered gravity when an apple fell on his head?", "gold_answer": "Isaac Newton", "common_hallucination": "Galileo Galilei"},
    {"question": "What is the capital of Japan?", "gold_answer": "Tokyo", "common_hallucination": "Kyoto"},
    {"question": "How many states are in the United States?", "gold_answer": "50", "common_hallucination": "52"},
    {"question": "Who is the author of 'Harry Potter'?", "gold_answer": "J.K. Rowling", "common_hallucination": "Stephenie Meyer"},
    {"question": "What is the fastest land animal?", "gold_answer": "Cheetah", "common_hallucination": "Lion"},
    {"question": "In which country was the Great Pyramid of Giza built?", "gold_answer": "Egypt", "common_hallucination": "Mexico"},
    {"question": "What is the smallest prime number?", "gold_answer": "2", "common_hallucination": "1"},
    {"question": "Which planet is closest to the Sun?", "gold_answer": "Mercury", "common_hallucination": "Venus"},
    {"question": "Who painted the Sistine Chapel ceiling?", "gold_answer": "Michelangelo", "common_hallucination": "Leonardo da Vinci"},
    {"question": "What is the currency of the United Kingdom?", "gold_answer": "Pound Sterling", "common_hallucination": "Euro"},
    {"question": "How many colors are in a rainbow?", "gold_answer": "7", "common_hallucination": "6 or 8"},
    {"question": "What is the capital of Italy?", "gold_answer": "Rome", "common_hallucination": "Milan"},
    {"question": "Which organ in the human body pumps blood?", "gold_answer": "Heart", "common_hallucination": "Liver"},
    {"question": "Who was the first man to walk on the moon?", "gold_answer": "Neil Armstrong", "common_hallucination": "Buzz Aldrin"},
    {"question": "What is the chemical formula for water?", "gold_answer": "H2O", "common_hallucination": "HO2"},
    {"question": "Which country is home to the Kangaroo?", "gold_answer": "Australia", "common_hallucination": "New Zealand"},
    {"question": "What is the largest planet in our solar system?", "gold_answer": "Jupiter", "common_hallucination": "Saturn"},
    {"question": "Who wrote 'The Odyssey'?", "gold_answer": "Homer", "common_hallucination": "Virgil"},
    {"question": "What is the capital of Canada?", "gold_answer": "Ottawa", "common_hallucination": "Toronto"},
    {"question": "Which metal is liquid at room temperature?", "gold_answer": "Mercury", "common_hallucination": "Gallium"},
    {"question": "How many days are in a leap year?", "gold_answer": "366", "common_hallucination": "365"},
    {"question": "Who is the current monarch of the UK?", "gold_answer": "Charles III", "common_hallucination": "Elizabeth II"},
    {"question": "What is the hardest natural substance on Earth?", "gold_answer": "Diamond", "common_hallucination": "Steel"},
    {"question": "Which ocean is located between the Americas and Europe/Africa?", "gold_answer": "Atlantic Ocean", "common_hallucination": "Pacific Ocean"},
    {"question": "Who founded Microsoft?", "gold_answer": "Bill Gates", "common_hallucination": "Steve Jobs"},
    {"question": "What is the capital of Germany?", "gold_answer": "Berlin", "common_hallucination": "Munich"},
    {"question": "Which animal is known as the King of the Jungle?", "gold_answer": "Lion", "common_hallucination": "Tiger"},
    {"question": "How many players are on a standard soccer team on the field?", "gold_answer": "11", "common_hallucination": "10 or 12"},
    {"question": "Who was the first woman to win a Nobel Prize?", "gold_answer": "Marie Curie", "common_hallucination": "Rosalind Franklin"},
    {"question": "What is the main ingredient in hummus?", "gold_answer": "Chickpeas", "common_hallucination": "Lentils"},
    {"question": "Which city is known as the Big Apple?", "gold_answer": "New York City", "common_hallucination": "Los Angeles"},
    {"question": "What is the most spoken language in the world by total speakers?", "gold_answer": "English", "common_hallucination": "Mandarin"},
    {"question": "Who painted 'The Starry Night'?", "gold_answer": "Vincent van Gogh", "common_hallucination": "Claude Monet"},
    {"question": "What is the capital of Spain?", "gold_answer": "Madrid", "common_hallucination": "Barcelona"},
    {"question": "Which gas makes up the majority of Earth's atmosphere?", "gold_answer": "Nitrogen", "common_hallucination": "Oxygen"},
    {"question": "How many years are in a decade?", "gold_answer": "10", "common_hallucination": "100"},
    {"question": "Who wrote 'The Great Gatsby'?", "gold_answer": "F. Scott Fitzgerald", "common_hallucination": "Ernest Hemingway"},
]


# ─────────────────────────────────────────────
#  Baseline: Helpers for diverse model support (OpenVINO/Transformers)
# ─────────────────────────────────────────────

def universal_chat(model_obj, tokenizer, processor, msgs, max_new_tokens=256, sampling=True, temperature=0.3):
    """A helper to handle both .chat() (MiniCPM) and .generate() (OpenVINO/Standard Llama)."""
    if hasattr(model_obj, "chat"):
        return model_obj.chat(
            msgs=msgs,
            tokenizer=tokenizer,
            processor=processor,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
            temperature=temperature,
        )
    else:
        if hasattr(tokenizer, "apply_chat_template"):
            encoded = tokenizer.apply_chat_template(
                msgs, 
                return_tensors="pt", 
                add_generation_prompt=True,
                return_dict=True
            ).to(model_obj.device)
        else:
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in msgs]) + "\nassistant: "
            encoded = tokenizer(prompt, return_tensors="pt").to(model_obj.device)
        
        input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
        attention_mask = encoded.get("attention_mask") if isinstance(encoded, dict) else getattr(encoded, "attention_mask", None)

        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": sampling,
            "temperature": temperature,
            "pad_token_id": tokenizer.eos_token_id
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        output_ids = model_obj.generate(**gen_kwargs)
        return tokenizer.decode(
            output_ids[0][input_ids.shape[-1]:], 
            skip_special_tokens=True
        ).strip()


def single_llm_query(model_obj, tokenizer, processor, question: str) -> dict:
    t0 = time.time()
    msgs = [
        {"role": "system", "content": "You are a helpful, accurate assistant."},
        {"role": "user",   "content": question},
    ]
    with torch.no_grad():
        answer = universal_chat(model_obj, tokenizer, processor, msgs=msgs, temperature=0.05)
    return {"answer": answer.strip(), "latency_ms": (time.time() - t0) * 1000, "method": "single_llm"}


def self_consistency_query(model_obj, tokenizer, processor, question: str, n: int = 5) -> dict:
    answers = []
    t0 = time.time()
    for _ in range(n):
        msgs = [
            {"role": "system", "content": "You are a helpful, accurate assistant."},
            {"role": "user",   "content": question},
        ]
        with torch.no_grad():
            ans = universal_chat(model_obj, tokenizer, processor, msgs=msgs, temperature=0.05)
        answers.append(ans.strip())
    return {"answer": max(set(answers), key=answers.count), "latency_ms": (time.time() - t0) * 1000, "method": "self_consistency"}


def check_correctness(answer, gold, correct_list, encoder=None):
    """Hybrid match: Exact substring match OR Cosine Similarity > 0.75."""
    targets = [gold] + [c for c in correct_list if c]
    for t in targets:
        if t.lower() in answer.lower():
            return True
    if encoder is not None and targets:
        from sklearn.metrics.pairwise import cosine_similarity
        gen_emb = encoder.encode([answer])
        tgt_emb = encoder.encode(targets)
        sims = cosine_similarity(gen_emb, tgt_emb)
        if sims.max() >= 0.75:
            return True
    return False


def print_separator(char="-", width=60):
    print(char * width)


def print_result_row(method: str, answer: str, gold: str, latency: float, confidence: float = None, is_correct: bool = False):
    correct = "✓" if is_correct else "✗"
    conf_str = f"  conf={confidence:.2f}" if confidence is not None else ""
    print(f"  [{correct}] {method:<22} {latency:>7.0f}ms{conf_str}")
    print(f"       Answer: {answer[:120]}{'...' if len(answer) > 120 else ''}")


# ─────────────────────────────────────────────
#  Storage & Persistence Helpers
# ─────────────────────────────────────────────

def get_storage_paths():
    """Determine where to save progress and summaries based on environment."""
    if os.path.exists("/content/drive/MyDrive"):
        drive_path = "/content/drive/MyDrive/BeeConsensus"
        if not os.path.exists(drive_path): os.makedirs(drive_path)
        checkpoint_file = os.path.join(drive_path, "benchmark_progress.csv")
        summary_file = os.path.join(drive_path, "benchmark_summary.txt")
    elif os.path.exists("/kaggle/working"):
        drive_path = "/kaggle/working"
        checkpoint_file = os.path.join(drive_path, "benchmark_progress.csv")
        summary_file = os.path.join(drive_path, "benchmark_summary.txt")
    else:
        drive_path = "."
        checkpoint_file = "benchmark_progress.csv"
        summary_file = "benchmark_summary.txt"
    return checkpoint_file, summary_file


def load_checkpoint_results(checkpoint_file):
    """Load results dictionary from a CSV checkpoint."""
    results = {
        "single_llm":        {"correct": 0, "latency": []},
        "self_consistency":  {"correct": 0, "latency": []},
        "beeconsensus":      {"correct": 0, "latency": [], "confidence": []},
    }
    processed_questions = set()
    if os.path.exists(checkpoint_file):
        try:
            import pandas as pd
            df_check = pd.read_csv(checkpoint_file)
            processed_questions = set(df_check["Question"].tolist())
            for _, row in df_check.iterrows():
                if "Single LLM Correct" in row:
                    results["single_llm"]["correct"] += int(row["Single LLM Correct"])
                    results["single_llm"]["latency"].append(row["Single LLM Latency"])
                if "Self-consistency Correct" in row:
                    results["self_consistency"]["correct"] += int(row["Self-consistency Correct"])
                    results["self_consistency"]["latency"].append(row["Self-consistency Latency"])
                if "BeeConsensus Correct" in row:
                    results["beeconsensus"]["correct"] += int(row["BeeConsensus Correct"])
                    results["beeconsensus"]["latency"].append(row["BeeConsensus Latency"])
                    results["beeconsensus"]["confidence"].append(row["BeeConsensus Confidence"])
        except Exception as e:
            print(f"[WARNING] Could not load checkpoint: {e}")
    return results, processed_questions


def print_and_save_summary(results, summary_file):
    """Calculate, print, and save the final evaluation report."""
    processed_count = len(results["beeconsensus"]["latency"])
    summary_lines = [
        "\n" + "="*60,
        f"  SUMMARY (TruthfulQA) - Persistent Report",
        "="*60,
        f"  Questions Processed: {processed_count}",
        f"  {'Method':<22} {'Accuracy':>10} {'Avg Latency':>14} {'Avg Conf':>10}",
        "-" * 60
    ]
    for method, data in results.items():
        acc     = (data["correct"] / processed_count * 100) if processed_count > 0 else 0
        avg_lat = mean(data["latency"]) if data["latency"] else 0
        conf_str = f"{mean(data['confidence']):.3f}" if data.get("confidence") else "  N/A "
        summary_lines.append(f"  {method:<22} {acc:>9.1f}%  {avg_lat:>10.0f} ms  {conf_str:>8}")
    summary_lines.append("\n  Evaluation complete.\n")
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    try:
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary_text)
        print(f"[INFO] Final results saved to: {summary_file}")
    except Exception as e:
        print(f"[ERROR] Could not save summary file: {e}")
    return summary_text


# ─────────────────────────────────────────────
#  Evaluation Modes
# ─────────────────────────────────────────────

def run_demo(bee: BeeConsensus):
    print("\n" + "="*60)
    print("  BeeConsensus — Demo Evaluation (Local Model)")
    print("="*60)
    results = {"single_llm": {"correct": 0, "latency": []}, "self_consistency": {"correct": 0, "latency": []}, "beeconsensus": {"correct": 0, "latency": [], "confidence": []}}
    for i, item in enumerate(DEMO_QUESTIONS[:10], 1):
        q, gold = item["question"], item["gold_answer"]
        print(f"\nQ{i}: {q}")
        r1 = single_llm_query(bee.model, bee.tokenizer, bee.processor, q)
        results["single_llm"]["correct"] += int(check_correctness(r1["answer"], gold, [], bee.encoder))
        r2 = self_consistency_query(bee.model, bee.tokenizer, bee.processor, q, n=len(bee.personas))
        results["self_consistency"]["correct"] += int(check_correctness(r2["answer"], gold, [], bee.encoder))
        t0 = time.time()
        res = bee.query(q)
        results["beeconsensus"]["correct"] += int(check_correctness(res.final_answer, gold, [], bee.encoder))
        results["beeconsensus"]["latency"].append((time.time() - t0) * 1000)
        results["beeconsensus"]["confidence"].append(res.confidence_score)
    print_and_save_summary(results, "demo_summary.txt")


def run_truthfulqa(bee: BeeConsensus, dataset_name: str, limit: int = 10):
    print("\n" + "="*60)
    print(f"  BeeConsensus — TruthfulQA Evaluation")
    print(f"  Dataset: {dataset_name} | Limit: {limit}")
    print("="*60)

    try:
        ds = load_dataset(dataset_name, "generation", split="train")
    except:
        ds = load_dataset(dataset_name, split="train")

    if limit and limit < len(ds):
        ds = ds.select(range(limit))

    checkpoint_file, summary_file = get_storage_paths()
    results, processed_questions = load_checkpoint_results(checkpoint_file)
    
    model_obj, tokenizer, processor = bee.model, bee.tokenizer, getattr(bee, "processor", None)

    for i, item in enumerate(ds, 1):
        q, gold = item["Question"], item["Best Answer"]
        if q in processed_questions:
            continue
            
        correct_list = item.get("Correct Answers", "").split(";")
        print(f"\nQ{i}: {q}")
        
        r1 = single_llm_query(model_obj, tokenizer, processor, q)
        is_c1 = check_correctness(r1["answer"], gold, correct_list, bee.encoder)
        results["single_llm"]["correct"] += int(is_c1)
        results["single_llm"]["latency"].append(r1["latency_ms"])
        print_result_row("Single LLM", r1["answer"], gold, r1["latency_ms"], is_correct=is_c1)

        r2 = self_consistency_query(model_obj, tokenizer, processor, q, n=len(bee.personas))
        is_c2 = check_correctness(r2["answer"], gold, correct_list, bee.encoder)
        results["self_consistency"]["correct"] += int(is_c2)
        results["self_consistency"]["latency"].append(r2["latency_ms"])
        print_result_row("Self-consistency", r2["answer"], gold, r2["latency_ms"], is_correct=is_c2)

        t0 = time.time()
        res = bee.query(q)
        elapsed = (time.time() - t0) * 1000
        is_cb = check_correctness(res.final_answer, gold, correct_list, bee.encoder)
        results["beeconsensus"]["correct"] += int(is_cb)
        results["beeconsensus"]["latency"].append(elapsed)
        results["beeconsensus"]["confidence"].append(res.confidence_score)
        print_result_row(f"BeeConsensus (r={res.rounds})", res.final_answer, gold, elapsed, confidence=res.confidence_score, is_correct=is_cb)

        try:
            file_exists = os.path.isfile(checkpoint_file)
            with open(checkpoint_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Question", "Target", "Single LLM Answer", "Single LLM Correct", "Single LLM Latency", "Self-consistency Answer", "Self-consistency Correct", "Self-consistency Latency", "BeeConsensus Answer", "BeeConsensus Correct", "BeeConsensus Latency", "BeeConsensus Confidence"])
                writer.writerow([q, gold, r1["answer"], int(is_c1), r1["latency_ms"], r2["answer"], int(is_c2), r2["latency_ms"], res.final_answer, int(is_cb), elapsed, res.confidence_score])
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint: {e}")

        cur_n = len(results["beeconsensus"]["latency"])
        print(f"     >> [LIVE ACCURACY] Bee: {(results['beeconsensus']['correct']/cur_n*100):.1f}% | Self-Con: {(results['self_consistency']['correct']/cur_n*100):.1f}% | Single: {(results['single_llm']['correct']/cur_n*100):.1f}% ({cur_n} questions)")

    print_and_save_summary(results, summary_file)
    return results


def main():
    parser = argparse.ArgumentParser(description="BeeConsensus Evaluation")
    parser.add_argument("--mode",      choices=["demo", "truthfulqa"], default="demo")
    parser.add_argument("--model_id",  default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--quorum",    type=float, default=0.60)
    parser.add_argument("--openvino",  action="store_true")
    parser.add_argument("--dataset",   default="domenicrosati/TruthfulQA")
    parser.add_argument("--limit",     type=int, default=5)
    parser.add_argument("--hf_token",  default=None)
    args = parser.parse_args()

    if args.mode == "truthfulqa":
        checkpoint_file, summary_file = get_storage_paths()
        results, processed_questions = load_checkpoint_results(checkpoint_file)
        total_needed = args.limit if args.limit > 0 else 817
        if len(processed_questions) >= total_needed and len(processed_questions) > 0:
            print(f"\n[INFO] All {len(processed_questions)} questions already finished!")
            print_and_save_summary(results, summary_file)
            return

    bee = BeeConsensus(model_id=args.model_id, quorum_threshold=args.quorum, verbose=True, use_openvino=args.openvino, hf_token=args.hf_token)
    if args.mode == "demo": run_demo(bee)
    elif args.mode == "truthfulqa": run_truthfulqa(bee, args.dataset, args.limit)

if __name__ == "__main__":
    main()
