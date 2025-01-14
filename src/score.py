import torch
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from sacrebleu.metrics import BLEU
import Levenshtein
from jiwer import wer, cer
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import logging
from tqdm import tqdm
import nltk
import os

# Download required NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricsCalculator:
    def __init__(self):
        logger.info("Initializing metrics calculator...")
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu = BLEU()
        
        # Initialize semantic model
        logger.info("Loading semantic model...")
        self.semantic_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.semantic_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        if torch.cuda.is_available():
            self.semantic_model = self.semantic_model.to('cuda')
        self.semantic_model.eval()

    def compute_rouge(self, pred: str, ref: str) -> dict:
        scores = self.rouge_scorer.score(pred, ref)
        return {
            'rouge1_f1': scores['rouge1'].fmeasure,
            'rouge2_f1': scores['rouge2'].fmeasure,
            'rougeL_f1': scores['rougeL'].fmeasure
        }

    def compute_meteor(self, pred: str, ref: str) -> float:
        return meteor_score([ref.split()], pred.split())

    def compute_edit_distance(self, pred: str, ref: str) -> dict:
        distance = Levenshtein.distance(pred, ref)
        max_len = max(len(pred), len(ref))
        normalized_score = 1 - (distance / max_len) if max_len > 0 else 0
        return {
            'edit_distance': distance,
            'normalized_edit_distance': normalized_score
        }

    def compute_cdm(self, pred: str, ref: str) -> float:
        pred_chars = set(pred)
        ref_chars = set(ref)
        if not ref_chars:
            return 0.0
        matches = pred_chars.intersection(ref_chars)
        return len(matches) / len(ref_chars)

    def compute_semantic_similarity(self, pred: str, ref: str) -> float:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        inputs = self.semantic_tokenizer([pred, ref], padding=True, truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = self.semantic_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        similarity = cosine_similarity(
            embeddings[0].cpu().reshape(1, -1), 
            embeddings[1].cpu().reshape(1, -1)
        )[0][0]
        return float(similarity)

    def compute_structural_similarity(self, pred: str, ref: str) -> float:
        structural_elements = [
            '\\begin', '\\end', '{', '}', '\\frac', '\\sqrt', 
            '\\sum', '\\prod', '\\int', '\\limits', '_', '^'
        ]
        
        def count_elements(text):
            return {elem: text.count(elem) for elem in structural_elements}
        
        pred_counts = count_elements(pred)
        ref_counts = count_elements(ref)
        
        total_elements = sum(ref_counts.values())
        if total_elements == 0:
            return 0.0
            
        matching_elements = sum(min(pred_counts[elem], ref_counts[elem]) 
                              for elem in structural_elements)
        
        return matching_elements / total_elements

def calculate_metrics_from_files(pred_file: str, ref_file: str, output_file: str):
    logger.info("Starting metrics calculation...")
    
    # Read files
    with open(pred_file, 'r') as f:
        predictions = [line.strip() for line in f.readlines()]
    with open(ref_file, 'r') as f:
        references = [line.strip() for line in f.readlines()]
    
    if len(predictions) != len(references):
        raise ValueError(f"Number of predictions ({len(predictions)}) does not match "
                        f"number of references ({len(references)})")
    
    calculator = MetricsCalculator()
    metrics = {
        'rouge1_f1': [], 'rouge2_f1': [], 'rougeL_f1': [],
        'meteor': [], 'edit_distance': [], 'normalized_edit_distance': [],
        'cdm': [], 'semantic_similarity': [], 'structural_similarity': [],
        'bleu': [], 'wer': [], 'cer': []
    }
    
    # Calculate metrics for each pair
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="Computing metrics"):
        # ROUGE scores
        rouge_scores = calculator.compute_rouge(pred, ref)
        for key, value in rouge_scores.items():
            metrics[key].append(value)
        
        # METEOR score
        metrics['meteor'].append(calculator.compute_meteor(pred, ref))
        
        # Edit distance
        edit_scores = calculator.compute_edit_distance(pred, ref)
        metrics['edit_distance'].append(edit_scores['edit_distance'])
        metrics['normalized_edit_distance'].append(edit_scores['normalized_edit_distance'])
        
        # CDM score
        metrics['cdm'].append(calculator.compute_cdm(pred, ref))
        
        # Semantic similarity
        metrics['semantic_similarity'].append(calculator.compute_semantic_similarity(pred, ref))
        
        # Structural similarity
        metrics['structural_similarity'].append(calculator.compute_structural_similarity(pred, ref))
        
        # BLEU score (for individual examples)
        bleu_score = calculator.bleu.sentence_score(pred, [ref]).score
        metrics['bleu'].append(bleu_score)
        
        # WER and CER
        metrics['wer'].append(wer(ref, pred))
        metrics['cer'].append(cer(ref, pred))
    
    # Calculate corpus-level BLEU
    corpus_bleu = calculator.bleu.corpus_score(predictions, [[ref] for ref in references]).score
    
    # Calculate average scores
    average_metrics = {
        'corpus_bleu': corpus_bleu,
        'average_scores': {
            metric: np.mean(scores) for metric, scores in metrics.items()
        },
        'std_scores': {
            metric: np.std(scores) for metric, scores in metrics.items()
        }
    }
    
    # Save results
    logger.info("Saving results...")
    with open(output_file, 'w') as f:
        json.dump(average_metrics, f, indent=2)
    
    # Print summary
    logger.info("\nMetrics Summary:")
    for metric, value in average_metrics['average_scores'].items():
        logger.info(f"{metric}: {value:.4f} ± {average_metrics['std_scores'][metric]:.4f}")
    
    return average_metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate metrics from prediction and reference files')
    parser.add_argument('--pred_file', type=str, default='predictions.txt',
                        help='Path to predictions file')
    parser.add_argument('--ref_file', type=str, default='references.txt',
                        help='Path to references file')
    parser.add_argument('--output_file', type=str, default='metrics_results.json',
                        help='Path to save metrics results')
    
    args = parser.parse_args()
    
    calculate_metrics_from_files(args.pred_file, args.ref_file, args.output_file)