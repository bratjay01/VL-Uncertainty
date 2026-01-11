# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 18:58:37 2026

@author: safaa.moallim
"""

"""
Hallucination Detection via Logical Inconsistency

This module implements a logic-based approach for detecting potential hallucinations
in vision–language models (LVLMs) by measuring self-contradictions under probing.

Given an input image and question, the method operates as follows:cd

0) The system first classifies the question as either:
   - FREE_FORM: open-ended questions requesting a general description or explanation.
   - SPECIFIC: questions requesting a specific visual detail (e.g., count, existence,
     attribute, or relation).

1) The LVLM is prompted to answer the question in sentence form.
   - For FREE_FORM questions, the answer may contain multiple sentences.
   - For SPECIFIC questions, the answer is treated as a single claim sentence.

2) Each claim sentence is treated independently and used to generate verification
   questions that probe different logical aspects of the claim, such as:
   existence, attributes, relations, and counting.

3) The LVLM is queried again with these verification questions, and its responses
   are collected as additional statements.

4) A language model (LLM) is used to judge whether pairs of collected statements
   logically contradict each other.

5) The frequency of detected contradictions is aggregated into an inconsistency
   score, which serves as a signal for potential hallucination.

Intuition:
Hallucinated content tends to produce unstable or self-contradictory responses
when the model is probed with logically related questions, whereas grounded content
remains logically consistent under such probing.
"""



import itertools
import re
from typing import Dict, List, Tuple, Optional

class LogicalInconsistencyCheck:
    """
    New structure (Jan 2026):
      - Two question types: FREE_FORM, VQA_STYLE
      - FREE_FORM: LVLM answer (multi-sentence) -> split -> per-sentence verification
      - VQA_STYLE: LVLM answer (single sentence) -> verification
      - Shared evaluation: pairwise contradiction checks -> simple inconsistency rate

    Assumptions:
      - self.lvlm.generate(image, prompt, temperature) -> str
      - self.llm.generate(prompt, temperature) -> str
    """

    def __init__(
        self,
        lvlm,
        llm,        
        num_questions_per_type: int = 1,
        inconsistency_threshold: float = 0.2,
        max_pairs: Optional[int] = None,  # optional cap to avoid O(n^2) blowup
        random_seed: int = 0,
        prompts: Optional[Dict[str, str]] = None,
        use_weighted_contradictions: bool = False,
        claim_weight: float = 2.0,
    ):
        self.lvlm = lvlm
        self.llm = llm
        self.num_questions_per_type = max(1, num_questions_per_type)
        self.inconsistency_threshold = inconsistency_threshold
        self.max_pairs = max_pairs
        self.random_seed = random_seed
        self.use_weighted_contradictions = use_weighted_contradictions
        self.claim_weight = claim_weight

        # Default prompts (format-robust, no JSON required)
        default_prompts = {

            "LLM_CLASSIFY_QUESTION_TYPE": (
                "Classify the following image question.\n"
                "If it asks for a general description or explanation, answer: FREE_FORM.\n"
                "If it asks for a specific visual detail (count, presence, attribute, relation), answer: SPECIFIC.\n"
                "Question: {question}\n"
                "Answer with only one word: FREE_FORM or SPECIFIC."
            ),
        
            "LVLM_SENTENCE_ANSWER": (
                "{question}\n"
                "Answer in one full sentence describing the visible evidence."
            ),
            
        
            "LLM_GEN_EXISTENCE_Q": (
                "Given the following statement about an image, write one yes-or-no question "
                "that checks whether something exists in the image.\n"
                "Statement: {claim_sentence}\n"
                "Question:"
            ),
        
            "LLM_GEN_ATTRIBUTE_Q": (
                "Given the following statement about an image, write one yes-or-no question "
                "that checks an attribute of something mentioned.\n"
                "Statement: {claim_sentence}\n"
                "Question:"
            ),
        
            "LLM_GEN_RELATION_Q": (
                "Given the following statement about an image, write one yes-or-no question "
                "that checks a spatial or relational fact between things mentioned.\n"
                "If no relation is mentioned, output 'NONE'.\n"
                "Statement: {claim_sentence}\n"
                "Question:"
            ),
        
            "LLM_GEN_COUNTING_Q": (
                "Given the following statement about an image, write one counting question "
                "related to the number of things mentioned.\n"
                "Statement: {claim_sentence}\n"
                "Question:"
            ),
        
            "LLM_CONTRADICTION_JUDGE": (
                "Do these two statements logically contradict each other?\n"
                "Answer only with CONTRADICT or NOT_CONTRADICT.\n"
                "S1: {s1}\n"
                "S2: {s2}"
            ),
        }


        self.prompts = default_prompts if prompts is None else {**default_prompts, **prompts}

    # ----------------------------
    # Public entry point
    # ----------------------------
    def process_sample(self, sample: Dict) -> Dict:
        """
        sample must contain:
          - sample['img']
          - sample['question']

        Returns a dict with:
          - initial_answer
          - claim_sentences
          - typed_questions (per claim)
          - verification_answers (per claim)
          - statement_pool
          - contradiction_pairs_count
          - total_pairs
          - inconsistency_rate
          - flag_inconsistent
        """
        print("----" * 30)
        print("Start of Logical Inconsistency Check (NEW STRUCTURE)")
        print("----" * 30)

        
        qtype = self._infer_question_type(sample["question"])
        

        # 1) LVLM initial answer (sentence(s))
        initial_answer = self._lvlm_answer_in_sentence(sample["img"], sample["question"], temp=0.1)
        print(f"- LVLM initial answer (sentence-style): {initial_answer}")

        # 2) Extract claim sentences depending on type
        if qtype == "FREE_FORM":
            claim_sentences = self._split_into_sentences(initial_answer)
        elif qtype == "VQA_STYLE":
            # Treat the whole sentence as one claim
            claim_sentences = [initial_answer.strip()] if initial_answer.strip() else []
        else:
            raise ValueError(f"Unknown question_type: {qtype}")

        print(f"- Extracted {len(claim_sentences)} claim sentence(s) from LVLM answer.")
        for i, s in enumerate(claim_sentences[:10]):
            print(f"  Claim[{i}]: {s}")
            pass

        # 3) For each claim, generate typed verification questions (LLM)
        typed_questions_per_claim = []
        for claim in claim_sentences:
            typed_qs = self._generate_typed_verification_questions(
                claim_sentence=claim,
                temp=0.1,
            )
            typed_questions_per_claim.append(typed_qs)
            
            
        

        # 4) Ask LVLM to answer each generated question in one sentence
        verification_answers_per_claim = []
        for typed_qs in typed_questions_per_claim:
            answers = {}
            for qtype_key, qlist in typed_qs.items():
                answers[qtype_key] = []
                for q in qlist:
                    ans = self._lvlm_answer_in_sentence(sample["img"], q, temp=0.7)
                    answers[qtype_key].append((q, ans))
            verification_answers_per_claim.append(answers)
            
            
        self.print_claims_questions_answers(claim_sentences, typed_questions_per_claim, verification_answers_per_claim)

        # 5) Build statement pool for contradiction checking
        statement_pools = self._build_statement_pools(
            claim_sentences=claim_sentences,
            verification_answers_per_claim=verification_answers_per_claim
        )

    




        # 6) Pairwise contradiction checks -> Simple inconsistency rate (PER CLAIM)
        all_inconsistency_rates = []
        all_pair_counts = []
        all_contradiction_counts = []
        
        # print("-------- Statement Pools (per claim) --------")
        
        for idx, pool in enumerate(statement_pools):
            print(f"\nClaim Pool {idx}:")
            print(f"- Pool size: {len(pool['statements'])}")
            for i, s in enumerate(pool['statements'][:12]):
                print(f"  P{idx}[{i}]: {s}")
        
            C, P_total, I = self._simple_inconsistency_rate(pool)
        
            print(f"- Contradictions: {C} / {P_total}")
            print(f"- Inconsistency rate (I_{idx}) = {I:.4f}")
        
            all_inconsistency_rates.append(I)
            all_pair_counts.append(P_total)
            all_contradiction_counts.append(C)




        # ---- Final inconsistency aggregation ----

        total_contradictions = sum(all_contradiction_counts)
        total_pairs = sum(all_pair_counts)
        
        if total_pairs > 0:
            inconsistency_rate = total_contradictions / total_pairs
        else:
            inconsistency_rate = 0.0
  


        # Final inconsistency decision
        flag_inconsistent = inconsistency_rate >= self.inconsistency_threshold
        
        print("-------- Final Inconsistency Summary --------")
        print(f"- Total contradictions: {total_contradictions}")
        print(f"- Total pairs checked: {total_pairs}")
        print(f"- Final inconsistency rate I = {inconsistency_rate:.4f}")
        print(f"- Threshold τ = {self.inconsistency_threshold}")
        print(f"- Final flag: {'INCONSISTENT' if flag_inconsistent else 'CONSISTENT'}")
        print("----" * 30)
        
        return {
            "question_type": qtype,
            "initial_answer": initial_answer,
            "claim_sentences": claim_sentences,
            "typed_questions_per_claim": typed_questions_per_claim,
            "verification_answers_per_claim": verification_answers_per_claim,
            "statement_pools": statement_pools,
            "total_contradictions": total_contradictions,
            "total_pairs": total_pairs,
            "inconsistency_rate": inconsistency_rate,
            "flag_inconsistent": flag_inconsistent,
        }



    # ----------------------------
    # Core helpers
    # ----------------------------
    
    def print_claims_questions_answers(self, claim_sentences, typed_questions_per_claim, verification_answers_per_claim):
        
        # ---- DEBUG: Inspect verification questions and answers for all claims ----
        print("\n======== DEBUG: Verification Questions and Answers ========")
        
        for claim_idx, claim in enumerate(claim_sentences):
            print(f"\n--- Claim {claim_idx} ---")
            print(f"Sentence: {claim}")
        
            q_dict = typed_questions_per_claim[claim_idx]
            a_dict = verification_answers_per_claim[claim_idx]
        
            for q_type in ["existence", "attribute", "relation", "counting"]:
                qs = q_dict.get(q_type, [])
                ans = a_dict.get(q_type, [])
        
                if len(qs) == 0:
                    continue
        
                print(f"\n  [{q_type.upper()}]")
                for i, q in enumerate(qs):
                    print(f"    Q{i}: {q}")
                    if i < len(ans):
                        print(f"    A{i}: {ans[i][1]}")
                    else:
                        print(f"    A{i}: <NO ANSWER>")


    
    
    def _infer_question_type(self, question: str) -> str:
        """
        Decide whether a question is FREE_FORM or SPECIFIC using an LLM.
        FREE_FORM: general description or explanation.
        SPECIFIC: asks for a specific visual detail (count, existence, attribute, relation).
        """
    
        prompt = self.prompts["LLM_CLASSIFY_QUESTION_TYPE"].format(
            question=question
        )
    
        raw = self.llm.generate(prompt, 0.1)
        raw = raw.strip().upper()
    
        # Robust parsing for small models
        if "FREE" in raw:
            return "FREE_FORM"
        if "SPECIFIC" in raw:
            return "VQA_STYLE"   # internally we still use VQA_STYLE naming
    
        # Fallback (conservative): treat as FREE_FORM
        return "FREE_FORM"


    def _lvlm_answer_in_sentence(self, img, question: str, temp: float = 0.1) -> str:
        prompt = self.prompts["LVLM_SENTENCE_ANSWER"].format(question=question)
        return self.lvlm.generate(img, prompt, temp)

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Simple sentence splitter (robust enough to start).
        You can replace with nltk/spacy later.
        """
        if not text:
            return []
        # Normalize whitespace
        t = re.sub(r"\s+", " ", text.strip())
        # Split on ., ?, ! but keep meaningful fragments
        parts = re.split(r"(?<=[.!?])\s+", t)
        # Filter tiny fragments
        sentences = [p.strip() for p in parts if len(p.strip()) > 3]
        return sentences

    def _generate_typed_verification_questions(
        self,
        claim_sentence: str,
        temp: float = 0.1,
    ) -> Dict[str, List[str]]:
        """
        Generate verification questions from a claim sentence using
        separate short prompts for each question type.
        """
    
        questions = {
            "existence": [],
            "attribute": [],
            "relation": [],
            "counting": []
        }
    
        # --- Existence question ---
        prompt = self.prompts["LLM_GEN_EXISTENCE_Q"].format(
            claim_sentence=claim_sentence
        )
        q = self.llm.generate(prompt, temp).strip()
        if q:
            questions["existence"].append(q)
    
        # --- Attribute question ---
        prompt = self.prompts["LLM_GEN_ATTRIBUTE_Q"].format(
            claim_sentence=claim_sentence
        )
        q = self.llm.generate(prompt, temp).strip()
        if q:
            questions["attribute"].append(q)
    
        # --- Relation question ---
        prompt = self.prompts["LLM_GEN_RELATION_Q"].format(
            claim_sentence=claim_sentence
        )
        q = self.llm.generate(prompt, temp).strip()
        if q and q.upper() != "NONE":
            questions["relation"].append(q)
    
        # --- Counting question ---
        prompt = self.prompts["LLM_GEN_COUNTING_Q"].format(
            claim_sentence=claim_sentence
        )
        q = self.llm.generate(prompt, temp).strip()
        if q:
            questions["counting"].append(q)
    
        return questions



    
    
    
    def _build_statement_pools(
        self,
        claim_sentences: List[str],
        verification_answers_per_claim: List[Dict[str, List[Tuple[str, str]]]]
    ) -> List[Dict]:
        """
        Build one statement pool per claim sentence.
    
        Each pool contains:
          - the original claim sentence
          - the sentence answers to verification questions generated from that claim
          - the claim_idx (index of the original claim in the statements list)
    
        Returns:
          A list of pools, where each pool is a dict with 'statements' and 'claim_idx'.
        """
    
        assert len(claim_sentences) == len(verification_answers_per_claim), \
            "Mismatch between claim sentences and verification answers."
    
        pools = []
    
        for claim, answers_dict in zip(claim_sentences, verification_answers_per_claim):
            statements = []
    
            # Add the original claim sentence (always at index 0)
            claim = claim.strip()
            if claim:
                statements.append(claim)
    
            # Add verification answers for this claim only
            for _, q_ans_list in answers_dict.items():
                for _, ans in q_ans_list:
                    ans = (ans or "").strip()
                    if ans:
                        statements.append(ans)
    
            # Optional: de-duplicate within this pool
            statements = list(dict.fromkeys(statements))
    
            pools.append({
                'statements': statements,
                'claim_idx': 0  # The claim is always at index 0
            })
    
        return pools


    # ----------------------------
    # Evaluation: Simple inconsistency rate
    # ----------------------------
    def _simple_inconsistency_rate(self, pool_dict: Dict) -> Tuple[float, float, float]:
        """
        Compute weighted inconsistency rate:
          I = sum(weights * contradictions) / sum(weights)
        
        If use_weighted_contradictions=True:
          - Contradictions involving the main claim get weight=claim_weight (default 2.0)
          - Contradictions between verification statements get weight=1.0
        Otherwise:
          - All contradictions get equal weight (original behavior)
        
        Returns:
          (weighted_contradiction_count, total_weight, weighted_inconsistency_rate)
        """
        statements = pool_dict['statements']
        claim_idx = pool_dict['claim_idx']
        
        if len(statements) < 2:
            return 0.0, 0.0, 0.0

        pairs = list(itertools.combinations(range(len(statements)), 2))

        # Optional cap on number of pairs (for speed). If you want exact O(n^2), set max_pairs=None.
        if self.max_pairs is not None and len(pairs) > self.max_pairs:
            # deterministic subsample
            import random
            rnd = random.Random(self.random_seed)
            pairs = rnd.sample(pairs, self.max_pairs)

        total_weight = 0.0
        contradiction_weight = 0.0

        for i, j in pairs:
            s1, s2 = statements[i], statements[j]
            verdict = self._judge_contradiction(s1, s2, temp=0.1)
            
            # Calculate weight for this pair
            if self.use_weighted_contradictions:
                # If either statement is the main claim, apply claim_weight
                if i == claim_idx or j == claim_idx:
                    weight = self.claim_weight
                else:
                    weight = 1.0
            else:
                weight = 1.0
            
            total_weight += weight
            
            if verdict == "CONTRADICT":
                contradiction_weight += weight

        # Weighted inconsistency rate
        inconsistency_rate = (contradiction_weight / total_weight) if total_weight > 0 else 0.0
        
        return contradiction_weight, total_weight, inconsistency_rate

    def _judge_contradiction(self, s1: str, s2: str, temp: float = 0.1) -> str:
        prompt = self.prompts["LLM_CONTRADICTION_JUDGE"].format(s1=s1, s2=s2)
        out = self.llm.generate(prompt, temp).strip().upper()
        # Be robust to extra text
        if "CONTRADICT" in out and "NOT_CONTRADICT" not in out:
            return "CONTRADICT"
        if "NOT_CONTRADICT" in out:
            return "NOT_CONTRADICT"
        # fallback: conservative
        return "NOT_CONTRADICT"

