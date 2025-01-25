import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from evaluate import load

def tfidf(text, n_sentences=50):
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
    return "\n".join(sentences_summary)


def get_split(split_path):
    split = {}
    for split_name in ['train', 'valid', 'test']:
        with open(os.path.join(split_path, f"{split_name}.txt")) as f:
            split[split_name] = f.read().splitlines()
    return split

def main():
    split_path = "/datas/store162/takala/ami/dataset/split"
    dataset_path = "/datas/store162/takala/ami/dataset/"
    result_path = "/datas/store162/takala/ami/results/extractive/tfidf"
    
    split = get_split(split_path)
    rouge = load("rouge")
    
    for split_name in ['train', 'valid', 'test']:
        predictions = []
        references = []
        os.makedirs(os.path.join(result_path, split_name), exist_ok=True)
        for meeting in split[split_name]:
            with open(os.path.join(dataset_path, "dialog", split_name, f"{meeting}"), "r") as f:
                dialogs = f.read()
            summary = tfidf(dialogs)
            with open(os.path.join(dataset_path, "extractive", split_name, f"{meeting}"), "r") as f:
                extractive = f.read()
            with open(os.path.join(result_path, split_name, f"{meeting}"), "w") as f:
                f.write(summary)
            print(summary)
            predictions.append(summary)
            references.append(extractive)
            
        rouge_score = rouge.compute(predictions=predictions, references=references)
        print(split_name, "ROUGE:", rouge_score)
        
if __name__ == "__main__":
    main()
