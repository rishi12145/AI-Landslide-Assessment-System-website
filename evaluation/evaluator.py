import re
import collections
import math
from typing import Dict, Any, List, Set

class GeotechnicalEvaluator:
    """
    Computes evaluation metrics (BLEU, ROUGE, Exact Match, Semantic Similarity)
    for comparing generated reports against professional reference targets.
    """

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Cleans and tokenizes text into lowercased alphanumeric words."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text.split()

    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> List[str]:
        """Generates n-grams from a list of tokens."""
        return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def calculate_bleu(self, reference: str, candidate: str, max_n: int = 4) -> float:
        """
        Computes simplified sentence BLEU score using n-gram precision and brevity penalty.
        """
        ref_tokens = self._tokenize(reference)
        cand_tokens = self._tokenize(candidate)
        
        if not cand_tokens or not ref_tokens:
            return 0.0
            
        precisions = []
        for n in range(1, max_n + 1):
            ref_ngrams = collections.Counter(self._get_ngrams(ref_tokens, n))
            cand_ngrams = collections.Counter(self._get_ngrams(cand_tokens, n))
            
            if not cand_ngrams:
                precisions.append(0.0)
                continue
                
            shared = sum((cand_ngrams & ref_ngrams).values())
            total = sum(cand_ngrams.values())
            precisions.append(shared / total if total > 0 else 0.0)
            
        # Log-average precision
        if min(precisions) == 0.0:
            return 0.0
            
        weight = 1.0 / max_n
        log_avg = sum(weight * math.log(p) for p in precisions)
        score = math.exp(log_avg)
        
        # Brevity penalty
        c = len(cand_tokens)
        r = len(ref_tokens)
        if c > r:
            bp = 1.0
        else:
            bp = math.exp(1 - r / c) if c > 0 else 0.0
            
        return bp * score

    def calculate_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
        """
        ref_tokens = self._tokenize(reference)
        cand_tokens = self._tokenize(candidate)
        
        if not ref_tokens or not cand_tokens:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougel": 0.0}
            
        # ROUGE-1 and ROUGE-2 helper
        def ngram_f1(n: int) -> float:
            ref_ng = set(self._get_ngrams(ref_tokens, n))
            cand_ng = set(self._get_ngrams(cand_tokens, n))
            intersection = ref_ng.intersection(cand_ng)
            
            if not intersection:
                return 0.0
                
            precision = len(intersection) / len(cand_ng)
            recall = len(intersection) / len(ref_ng)
            return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # ROUGE-L helper (Longest Common Subsequence)
        def lcs(x: List[str], y: List[str]) -> int:
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]

        lcs_len = lcs(ref_tokens, cand_tokens)
        p_lcs = lcs_len / len(cand_tokens)
        r_lcs = lcs_len / len(ref_tokens)
        rouge_l = (2 * p_lcs * r_lcs) / (p_lcs + r_lcs) if (p_lcs + r_lcs) > 0 else 0.0

        return {
            "rouge1": ngram_f1(1),
            "rouge2": ngram_f1(2),
            "rougel": rouge_l
        }

    def calculate_exact_match(self, reference: str, candidate: str) -> float:
        """Returns 1.0 if candidate matches reference exactly (stripped), else 0.0."""
        return 1.0 if reference.strip() == candidate.strip() else 0.0

    def calculate_semantic_similarity(self, reference: str, candidate: str) -> float:
        """
        Calculates cosine similarity of word frequencies (Bag of Words / TF-IDF vectorizer fallback).
        """
        ref_tokens = self._tokenize(reference)
        cand_tokens = self._tokenize(candidate)
        
        if not ref_tokens or not cand_tokens:
            return 0.0
            
        ref_counts = collections.Counter(ref_tokens)
        cand_counts = collections.Counter(cand_tokens)
        
        all_words = set(ref_counts.keys()).union(set(cand_counts.keys()))
        
        # Calculate dot product and magnitudes
        dot_product = 0.0
        ref_mag = 0.0
        cand_mag = 0.0
        
        for word in all_words:
            ref_val = ref_counts.get(word, 0)
            cand_val = cand_counts.get(word, 0)
            
            dot_product += ref_val * cand_val
            ref_mag += ref_val ** 2
            cand_mag += cand_val ** 2
            
        if ref_mag == 0.0 or cand_mag == 0.0:
            return 0.0
            
        return dot_product / (math.sqrt(ref_mag) * math.sqrt(cand_mag))

    def evaluate_report(self, reference: str, candidate: str) -> Dict[str, Any]:
        """
        Aggregates all metric scores into a single diagnostic report comparison dictionary.
        """
        rouge = self.calculate_rouge(reference, candidate)
        bleu = self.calculate_bleu(reference, candidate)
        similarity = self.calculate_semantic_similarity(reference, candidate)
        exact = self.calculate_exact_match(reference, candidate)
        
        return {
            "bleu": round(bleu, 4),
            "rouge1": round(rouge["rouge1"], 4),
            "rouge2": round(rouge["rouge2"], 4),
            "rougel": round(rouge["rougel"], 4),
            "exact_match": exact,
            "cosine_semantic_similarity": round(similarity, 4)
        }
