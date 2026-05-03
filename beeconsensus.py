import os
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
try:
    from optimum.intel.openvino import OVModelForCausalLM
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BeeConsensus")


# ─────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────

class QuorumStatus(Enum):
    REACHED   = "quorum_reached"
    FAILED    = "quorum_failed"
    MAX_ROUNDS = "max_rounds_exceeded"


@dataclass
class AgentResponse:
    agent_id:   int
    persona:    str
    answer:     str
    latency_ms: float


@dataclass
class ConsensusResult:
    query:              str
    final_answer:       str
    confidence_score:   float        # 0.0 – 1.0
    quorum_status:      QuorumStatus
    rounds:             int
    cluster_size:       int
    total_agents:       int
    agent_responses:    list[AgentResponse] = field(default_factory=list)
    flagged_for_review: bool = False

    def to_dict(self) -> dict:
        return {
            "query":              self.query,
            "final_answer":       self.final_answer,
            "confidence_score":   round(self.confidence_score, 4),
            "quorum_status":      self.quorum_status.value,
            "rounds":             self.rounds,
            "cluster_size":       self.cluster_size,
            "total_agents":       self.total_agents,
            "flagged_for_review": self.flagged_for_review,
            "agent_responses": [
                {"agent_id": r.agent_id, "persona": r.persona,
                 "answer": r.answer, "latency_ms": round(r.latency_ms, 1)}
                for r in self.agent_responses
            ],
        }


# ─────────────────────────────────────────────
#  Scout agent personas  (the forager bees)
# ─────────────────────────────────────────────

AGENT_PERSONAS = [
    {
        "name": "Factual Specialist",
        "system_prompt": (
            "You are a factual specialist. Answer questions using only verified facts. "
            "If uncertain, you must explicitly state 'I have no comment' or 'I do not know'. "
            "Never guess or fabricate information. Be concise and precise."
        ),
    },
    {
        "name": "Cautious Reasoner",
        "system_prompt": (
            "You are a cautious reasoner. Approach every question skeptically, especially trick questions. "
            "If there is no definitive factual answer, reply with 'I have no comment'. "
            "Prefer admitting ignorance over guessing."
        ),
    },
    {
        "name": "Citation-Focused",
        "system_prompt": (
            "You are a citation-focused assistant. Answer questions based on established scientific or historical consensus. "
            "If a claim lacks strong evidence or is a common myth, debunk it or state 'I have no comment'. "
            "Do not perpetuate misconceptions."
        ),
    },
    {
        "name": "General Baseline",
        "system_prompt": (
            "You are a highly accurate assistant. Answer the question directly and clearly. "
            "You are penalized for hallucinating. If you do not know the exact truth, you must reply 'I have no comment'."
        ),
    },
    {
        "name": "Devil's Advocate",
        "system_prompt": (
            "You are a devil's advocate. Consider what common misconceptions people give to this question. "
            "Then provide the correct, accurate answer, explicitly avoiding those errors. "
            "If the question is a trick and has no factual basis, reply 'I have no comment'."
        ),
    },
]


# ─────────────────────────────────────────────
#  BeeConsensus framework
# ─────────────────────────────────────────────

class BeeConsensus:
    """
    Swarm-based LLM consensus system for hallucination reduction.

    Architecture:
        1. Scout agents  – N independent LLMs with distinct personas generate answers
        2. Dance floor   – Semantic similarity matrix + DBSCAN clustering
        3. Quorum gate   – Dominant cluster must reach Q% threshold
        4. Re-deliberate – If quorum fails, agents see each other's answers and retry
        5. Calibration   – Confidence = cluster cohesion × cluster dominance
    """

    def __init__(
        self,
        model_id:         str  = "mujtaba025/tiny-random-MiniCPM-o-2_6",
        embedder:         str  = "all-MiniLM-L6-v2",
        quorum_threshold: float = 0.60,   # fraction of agents required for consensus
        max_rounds:       int   = 2,
        review_threshold: float = 0.45,   # confidence below this → flagged
        dbscan_eps:       float = 0.30,   # cosine-distance neighbourhood radius
        dbscan_min:       int   = 2,      # min samples for a core point
        verbose:          bool  = True,
        device:           str   = "cpu",
        use_openvino:     bool  = False,
        hf_token:         Optional[str] = None
    ):
        self.model_id         = model_id
        self.quorum_threshold = quorum_threshold
        self.max_rounds       = max_rounds
        self.review_threshold = review_threshold
        self.dbscan_eps       = dbscan_eps
        self.dbscan_min       = dbscan_min
        self.verbose          = verbose
        self.personas         = AGENT_PERSONAS
        self.use_openvino     = use_openvino
        self.hf_token         = hf_token
        self.device           = device if torch.cuda.is_available() or "mps" in device else "cpu"

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"

        logger.info(f"Loading local model: {model_id} (OpenVINO={use_openvino}) on {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=self.hf_token, trust_remote_code=True)
        # Try to load processor for multimodal models (like MiniCPM), but keep it optional
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, token=self.hf_token, trust_remote_code=True)
        except Exception:
            self.processor = None

        if self.use_openvino:
            if not OPENVINO_AVAILABLE:
                raise ImportError("OpenVINO is not available. Please install optimum[openvino].")
            logger.info("Exporting/Loading model to OpenVINO format...")
            # OpenVINO device mapping
            # OpenVINO device preference: GPU (Intel iGPU/dGPU) > CPU
            # Forced to CPU as requested to prevent OpenCL memory crashes with 8B model
            ov_device = "CPU"
            logger.info(f"Using OpenVINO device: {ov_device}")
                
            self.model = OVModelForCausalLM.from_pretrained(
                model_id,
                export=True,
                trust_remote_code=True,
                device=ov_device,
                token=self.hf_token,
                load_in_4bit=True  # Significantly reduces RAM and system load
            )
        else:
            # Native PyTorch loading (Used in Google Colab)
            # We use BitsAndBytesConfig so the 8B model fits comfortably in a free 16GB T4 GPU.
            kwargs = {
                "trust_remote_code": True,
                "token": self.hf_token,
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
            }
            if self.device == "cuda":
                try:
                    from transformers import BitsAndBytesConfig
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    logger.info("Using BitsAndBytes 4-bit quantization for NVIDIA GPU.")
                except ImportError:
                    kwargs["load_in_4bit"] = True
                    logger.info("Using basic 4-bit quantization (BitsAndBytesConfig not found).")

            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            if self.device != "cuda":
                self.model = self.model.to(self.device)
        self.model.eval()

        # Sentence encoder (downloads once, ~80 MB)
        logger.info(f"Loading sentence encoder: {embedder}")
        self.encoder = SentenceTransformer(embedder)
        logger.info("BeeConsensus ready with local model.")

    # ── 1. Scout phase ────────────────────────────────────────────────────

    def _query_agent(
        self,
        agent_id:   int,
        persona:    dict,
        user_query: str,
        context:    Optional[str] = None,
    ) -> AgentResponse:
        """Query a single agent with its persona system prompt locally."""

        msgs = [{"role": "system", "content": persona["system_prompt"]}]

        if context:
            msgs.append({
                "role": "user",
                "content": (
                    f"Other agents have offered the following answers to this question:\n"
                    f"{context}\n\n"
                    f"Now provide your own best answer, taking these into account:\n{user_query}"
                ),
            })
        else:
            msgs.append({"role": "user", "content": user_query})

        t0 = time.time()
        
        # Use local chat method
        with torch.no_grad():
            if hasattr(self.model, "chat"):
                # Use MiniCPM/special chat method
                answer = self.model.chat(
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    max_new_tokens=512,
                    sampling=True,
                    temperature=0.05,
                )
            else:
                # OpenVINO or Standard Transformers (Llama)
                if hasattr(self.tokenizer, "apply_chat_template"):
                    encoded = self.tokenizer.apply_chat_template(
                        msgs, 
                        return_tensors="pt", 
                        add_generation_prompt=True,
                        return_dict=True
                    ).to(self.model.device)
                else:
                    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in msgs]) + "\nassistant: "
                    encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                
                input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
                attention_mask = encoded.get("attention_mask") if isinstance(encoded, dict) else getattr(encoded, "attention_mask", None)

                gen_kwargs = {
                    "input_ids": input_ids,
                    "max_new_tokens": 512,
                    "do_sample": True,
                    "temperature": 0.05,
                    "pad_token_id": self.tokenizer.eos_token_id
                }
                if attention_mask is not None:
                    gen_kwargs["attention_mask"] = attention_mask

                output_ids = self.model.generate(**gen_kwargs)
                
                # Decode only the new tokens
                answer = self.tokenizer.decode(
                    output_ids[0][input_ids.shape[-1]:], 
                    skip_special_tokens=True
                ).strip()
            
        latency_ms = (time.time() - t0) * 1000

        if self.verbose:
            logger.info(f"  Agent {agent_id} [{persona['name']}] → {len(answer)} chars "
                        f"({latency_ms:.0f} ms)")

        return AgentResponse(
            agent_id=agent_id,
            persona=persona["name"],
            answer=answer,
            latency_ms=latency_ms,
        )


    def _run_scouts(
        self,
        query:   str,
        context: Optional[str] = None,
    ) -> list[AgentResponse]:
        """Query all scout agents (sequentially to respect rate limits)."""
        responses = []
        for i, persona in enumerate(self.personas):
            resp = self._query_agent(i, persona, query, context)
            responses.append(resp)
        return responses

    # ── 2. Dance floor (semantic clustering) ──────────────────────────────

    def _cluster_responses(
        self, responses: list[AgentResponse]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Embed responses and cluster with DBSCAN on cosine distance.
        Returns (labels, embeddings).
        """
        answers    = [r.answer for r in responses]
        embeddings = self.encoder.encode(answers, normalize_embeddings=True)

        # DBSCAN uses euclidean by default; convert cosine sim to distance
        cos_sim  = cosine_similarity(embeddings)
        cos_dist = np.clip(1.0 - cos_sim, 0.0, 2.0)

        db     = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min, metric="precomputed")
        labels = db.fit_predict(cos_dist)

        if self.verbose:
            unique = set(labels)
            logger.info(f"  Clusters found: {unique}")

        return labels, embeddings

    # ── 3. Quorum gate ────────────────────────────────────────────────────

    def _check_quorum(
        self,
        labels:    np.ndarray,
        n_agents:  int,
    ) -> tuple[bool, int, int]:
        """
        Return (quorum_reached, winning_cluster_id, winning_cluster_size).
        Noise points (label == -1) are excluded from quorum counting.
        """
        non_noise = labels[labels != -1]
        if len(non_noise) == 0:
            return False, -1, 0

        unique, counts = np.unique(non_noise, return_counts=True)
        best_idx       = int(np.argmax(counts))
        winner_label   = int(unique[best_idx])
        winner_count   = int(counts[best_idx])

        fraction = winner_count / n_agents
        reached  = fraction >= self.quorum_threshold

        if self.verbose:
            logger.info(
                f"  Quorum check: cluster {winner_label} has {winner_count}/{n_agents} "
                f"agents ({fraction:.0%}) — threshold {self.quorum_threshold:.0%} "
                f"→ {'✓ REACHED' if reached else '✗ FAILED'}"
            )

        return reached, winner_label, winner_count

    # ── 4. Centroid answer selection ──────────────────────────────────────

    def _select_centroid_answer(
        self,
        responses:     list[AgentResponse],
        labels:        np.ndarray,
        embeddings:    np.ndarray,
        winner_label:  int,
    ) -> str:
        """Return the response closest to the centroid of the winning cluster."""
        mask      = labels == winner_label
        members   = np.where(mask)[0]
        cluster_e = embeddings[mask]
        centroid  = cluster_e.mean(axis=0)

        sims      = cosine_similarity([centroid], cluster_e)[0]
        best_local= int(np.argmax(sims))
        best_idx  = int(members[best_local])

        return responses[best_idx].answer

    # ── 5. Confidence calibration (waggle dance signal) ───────────────────

    def _compute_confidence(
        self,
        labels:       np.ndarray,
        embeddings:   np.ndarray,
        winner_label: int,
        n_agents:     int,
    ) -> float:
        """
        confidence = cluster_cohesion × cluster_dominance

        cohesion   = mean intra-cluster cosine similarity
        dominance  = winning cluster size / total agents
        """
        mask      = labels == winner_label
        cluster_e = embeddings[mask]
        n_cluster = mask.sum()

        if n_cluster < 2:
            cohesion = 0.5
        else:
            sim_matrix = cosine_similarity(cluster_e)
            # mean of upper triangle (excluding diagonal)
            upper = sim_matrix[np.triu_indices(n_cluster, k=1)]
            cohesion = float(np.mean(upper))

        dominance  = n_cluster / n_agents
        confidence = cohesion * dominance

        if self.verbose:
            logger.info(
                f"  Confidence: cohesion={cohesion:.3f} × "
                f"dominance={dominance:.3f} = {confidence:.3f}"
            )

        return float(confidence)

    # ── Main entry point ─────────────────────────────────────────────────

    def query(self, user_query: str) -> ConsensusResult:
        """
        Run BeeConsensus on a single query.

        Pipeline:
            Round 1 → Scout → Cluster → Quorum?
                YES  → Calibrate → Return result
                NO   → Re-deliberate (max max_rounds times)
            If max_rounds exceeded → return best-effort answer
        """
        logger.info(f"\n{'='*60}\nQuery: {user_query}\n{'='*60}")

        all_responses: list[AgentResponse] = []
        context: Optional[str] = None
        n = len(self.personas)

        for round_num in range(1, self.max_rounds + 2):  # +1 for initial round
            logger.info(f"\n── Round {round_num} ──")

            responses = self._run_scouts(user_query, context)
            all_responses = responses  # keep only latest round for clustering

            labels, embeddings = self._cluster_responses(responses)
            reached, winner_label, winner_count = self._check_quorum(labels, n)

            if reached:
                final_answer = self._select_centroid_answer(
                    responses, labels, embeddings, winner_label
                )
                confidence = self._compute_confidence(
                    labels, embeddings, winner_label, n
                )
                result = ConsensusResult(
                    query            = user_query,
                    final_answer     = final_answer,
                    confidence_score = confidence,
                    quorum_status    = QuorumStatus.REACHED,
                    rounds           = round_num,
                    cluster_size     = winner_count,
                    total_agents     = n,
                    agent_responses  = responses,
                    flagged_for_review = confidence < self.review_threshold,
                )
                logger.info(f"\n✓ Consensus reached in round {round_num}. "
                            f"Confidence: {confidence:.3f}")
                return result

            # Quorum not reached — build re-deliberation context
            if round_num <= self.max_rounds:
                logger.info("  Building re-deliberation context...")
                context = "\n".join(
                    f"[Agent {r.agent_id} – {r.persona}]: {r.answer}"
                    for r in responses
                )

        # Max rounds exceeded — fall back to most popular answer
        logger.warning("Max re-deliberation rounds exceeded. Returning best-effort answer.")

        # Recompute best cluster even without quorum
        labels, embeddings = self._cluster_responses(all_responses)
        non_noise = labels[labels != -1]

        if len(non_noise) > 0:
            unique, counts = np.unique(non_noise, return_counts=True)
            winner_label   = int(unique[np.argmax(counts)])
            winner_count   = int(np.max(counts))
        else:
            winner_label  = -1
            winner_count  = 0

        if winner_label != -1:
            final_answer = self._select_centroid_answer(
                all_responses, labels, embeddings, winner_label
            )
            confidence = self._compute_confidence(
                labels, embeddings, winner_label, n
            )
        else:
            # All agents are noise — pick first response
            final_answer = all_responses[0].answer
            confidence   = 0.1

        return ConsensusResult(
            query              = user_query,
            final_answer       = final_answer,
            confidence_score   = confidence,
            quorum_status      = QuorumStatus.MAX_ROUNDS,
            rounds             = self.max_rounds + 1,
            cluster_size       = winner_count,
            total_agents       = n,
            agent_responses    = all_responses,
            flagged_for_review = True,
        )
