# MSqrd at SemEval-2026 Task 9: Detecting Online Polarization

This repository contains the official implementation for the **MSqrd** submission to SemEval-2026 Task 9: *Detecting Multilingual, Multicultural and Multievent Online Polarization*. We provide a BERT-based framework to identify polarization across 22 languages and diverse social contexts.

---

## 🚀 Project Overview

The project addresses three distinct subtasks designed to capture the complexity of online discourse:

1.  **Subtask 1: Polarization Detection** – Binary classification (Polar vs. Non-polar).
2.  **Subtask 2: Polarization Type** – Multi-label classification (Political, Racial, Religious, Gender, etc.).
3.  **Subtask 3: Manifestation Identification** – Multi-label detection of expression (Stereotype, Vilification, Dehumanization, etc.).

---

## 📊 Results Summary

Our fine-tuned transformer models significantly outperformed the task baselines across most languages.

| Subtask | Best Model Configuration | Avg. Macro-F1 |
| :--- | :--- | :--- |
| **ST1** | XLM-R Large + Focal Loss | **78.6** |
| **ST2** | Twitter-XLM-RoBERTa + Asymmetric Loss (ASL) | **55.8** |
| **ST3** | Ensemble (XLM-R Large + mDeBERTa) | **44.6** |

> [!NOTE]
> We achieved particularly high performance in languages like Urdu, Hindi, Nepali, and Chinese.

---

## 🛠️ Technical Approach

### Model Architectures
*   **Transformer Backbone:** Primary use of **XLM-RoBERTa Large**, **mDeBERTa**, and **LaBSE** to handle multilingual embeddings.
*   **Domain Adaptation:** For ST2, we utilized `twitter-xlm-roberta-base-sentiment` to leverage domain-specific pre-training on social media data.
*   **Ensemble Methods:** ST3 employs an ensemble of `XLM-R-Large-Polarization Classifier` and mDeBERTa to balance precision and recall across minority labels.

### Optimization Strategies
*   **Focal Loss:** Implemented in ST1 to address class imbalance and focus on hard-to-classify instances.
*   **Asymmetric Loss (ASL):** Used in ST2 to balance hard positives and easy negatives by using separate parameters for each.
*   **Language-Aware Scaling:** Employed `WeightedRandomSampler` and per-language loss scaling for ST3 to manage the 18-language distribution shift.

---

## 📁 Dataset Statistics
The models were trained on a diverse dataset covering 22 languages, including low-resource languages such as Hausa and Amharic.

*   **Total Samples:** 77,368 (ST1/ST2) | 64,810 (ST3).
*   **Imbalance Ratios:** Ranges from 1:1.13 (ST1) to 1:3.23 (ST2).
  booktitle={Proceedings of the 20th International Workshop on Semantic Evaluation (SemEval-2026)},
  year={2026}
}
