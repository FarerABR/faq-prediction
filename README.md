# faq-prediction — STS vs LLM Comparison

This report is prepared as part of a bank FAQ prediction/classification project (FAQ Matching) using two approaches: **STS** and **LLM**.

Project goal: for each user question (in the `samples` sheet), predict an `idx` from the `faq` sheet and then compare it with the ground-truth label `gt_idx`.  
In this project, GPT-5.1-codex was used in a limited manner to correct and complete the code.

## 1) Evaluation Metrics

The main metric in this project is **Top-1**.

### Top-1 Accuracy and Top-1 Error

If for sample $i$ the ground-truth label is $y_i$ and the prediction is $\hat{y}_i$:

$$
{Accuracy} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[\hat{y}_i = y_i]
$$

$$
{Error} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[\hat{y}_i \ne y_i] = 1 - \text{Accuracy}
$$

### Top-K Accuracy (Hit@K / Acc@K)

If $R_i^{(K)}$ is the set/list of the top-$K$ predictions for sample $i$:

$$
{Acc@K} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[y_i \in R_i^{(K)}]
$$

---

## 2) Method 1 — STS (Embedding Similarity)

### General Idea

In STS, texts are converted into numerical vectors (embeddings), and then the closest FAQ question is selected using a similarity metric (cosine distance in this case).

### Model Used

- Multilingual embedding model suitable for Persian: `intfloat/multilingual-e5-base`

In E5, roles must be specified:

- FAQ questions as `passage: ...`
- User question as `query: ...`

### Persian Preprocessing

To reduce noise and increase semantic similarity, normalization was applied:

- Convert `ي` → `ی` and `ك` → `ک`
- Remove/convert some invisible characters such as `\u200c`, etc.
- Normalize spacing

### Similarity Computation and Prediction

Let the embedding of FAQ item $j$ be $e(f_j)$ and the embedding of sample $i$ be $e(q_i)$:

$$
s_{j,i} = \cos(e(f_j), e(q_i))
$$

Since `normalize_embeddings=True`, vectors are unit-normalized and cosine similarity equals the dot product.

Top-1 prediction:

$$
\hat{y}_i = \text{idx}\left(\arg\max_j s_{j,i}\right)
$$

Top-1 score:

$$
s_i = \max_j s_{j,i}
$$

### Outputs

The following columns were added to the `samples` sheet:

- `sts_idx`: predicted `idx`
- `sts_score`: Top-1 similarity score

---

## 3) Method 2 — LLM (Large Language Model) with Prompting

### General Idea

In the LLM approach, for each sample:

- The full FAQ list (with `idx` and question text) plus the user question is placed in the prompt.
- The model must select **exactly one** `idx`.

Since the number of FAQs is small (24 rows), context length is not an issue.

To test this method, two models and prompts in both Persian and English were used.

### Execution Infrastructure

Models were executed on a system with 48 GB RAM, 8 CPU cores, and an L4 GPU.

### Models

For comparison, at least two models were selected:

- `google/gemma-3-4b-it`
- `Qwen/Qwen2.5-7B-Instruct`

### Prompt and Output Format

To reduce parsing errors and increase stability:

- Output must be JSON only, in the following format:
```json
{ "idx": 12 }
```
- Generation parameters are set deterministically (e.g., `temperature=0`).

### Why `parse_ok`

The LLM may produce unparsable output (extra text, incomplete JSON, multiple idx values, etc.).  
Therefore, alongside accuracy, the parse success rate is also reported:

$$
{parse\_ok\_rate} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\text{parse ok}_i]
$$

### Challenges

The two mentioned models did not produce satisfactory outputs with a simple prompt, mainly due to output parsing issues.  
To address this, several different prompts were tested. Eventually, the best approach was using multiple examples or a few-shot technique.

With this method, both models reached `parse_rate = 1` in English.  
Additionally, a retry mechanism was added so that if a sample’s output was invalid, it would be re-evaluated by the model.

Given the results, this shows that **English prompts with examples** were the best option.

---

## 4) Results

### STS Results

- Top-1 Accuracy: **0.5914**
- Top-1 Error: **0.4086**
- Acc@3: **0.8387**

### LLM Results (per model)

EN - gemma-3-4b-it:

- Overall Accuracy (invalid outputs counted as errors): **0.7849**
- parse_ok_rate: **1.0**
- Error: **0.2150**
- Run time (ms/sample): **13.89**

EN - Qwen2.5-7B-Instruct:

- Overall Accuracy (invalid outputs counted as errors): **0.8064**
- parse_ok_rate: **1.0**
- Error: **0.1935**
- Run time (ms/sample): **40.63**

FA - gemma-3-4b-it:

- Overall Accuracy (invalid outputs counted as errors): **0.7311**
- parse_ok_rate: **1.0**
- Error: **0.2688**
- Run time (ms/sample): **13.68**

FA - Qwen2.5-7B-Instruct:

- Overall Accuracy (invalid outputs counted as errors): **0.6774**
- parse_ok_rate: **0.8924**
- Error: **0.3225**
- Run time (ms/sample): **44.44**

---

## 5) Model Comparison and Summary

Based on the results, LLMs achieved better performance overall, and in some cases their Top-1 error was comparable to the Top-3 error of the STS model.

Regarding language and LLMs, the Qwen model performed better in English. Meanwhile, the Gemma model produced solid results in both languages and had significantly faster inference per prompt compared to Qwen (approximately **2.9× faster**).

---
