import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from datasets import load_dataset
from rouge_score import rouge_scorer
from tqdm import tqdm

def extractive_summarization_tfidf(text, n_sentences=3):
    sentences = sent_tokenize(text)
    if len(sentences) <= n_sentences:
        return text

    word_tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
    processed_sentences = [" ".join(words) for words in word_tokenized_sentences]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_sentences)
    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
    ranked_sentences_indices = np.argsort(-sentence_scores)
    top_sentence_indices = sorted(ranked_sentences_indices[:n_sentences])
    sentences_summary = [sentences[idx] for idx in top_sentence_indices]
    return sentences_summary, " ".join(sentences_summary)

def evaluate_with_rouge(generated_summary, reference_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    return scores

def main():
    nltk.download('punkt')
    dataset = load_dataset("TakalaWang/AMI_WHISPER_ASR", split="train")
    
    all_rouge_scores = {
        "rouge1": [],
        "rouge2": [],
        "rougeL": []
    }
    for data in tqdm(dataset):
        transcript = data["transcript"]
        reference = data.get("summary", "")
        
        sentences_summary, summary = extractive_summarization_tfidf(transcript)
        print(sentences_summary)
        
        rouge_scores = evaluate_with_rouge(summary, reference)
        for key, value in rouge_scores.items():
            all_rouge_scores[key].append(value.fmeasure)
    print(
        f"rouge1: {np.mean(all_rouge_scores['rouge1']):.4f}",
        f"rouge2: {np.mean(all_rouge_scores['rouge2']):.4f}",
        f"rougeL: {np.mean(all_rouge_scores['rougeL']):.4f}",
    )
    
    
if __name__ == "__main__":
    main()
