# LectureMIND — Query Routing Evaluation Report

**Dataset:** `evaluation/datasets/cs162_lecture1_routing_12.json`  
**Sample Size (N):** 12  
**Overall Accuracy:** 100.0% (12/12)  

## 1. Per-Class Route Accuracy

| Route | Total | Correct | Accuracy |
|---|---:|---:|---:|
| `vector_only` | 5 | 5 | 100.0% |
| `graph_only` | 5 | 5 | 100.0% |
| `graph+vector` | 2 | 2 | 100.0% |

## 2. Confusion Matrix

| Expected \ Predicted | vector_only | graph_only | graph+vector |
|---|---:|---:|---:|
| **vector_only** | 5 | 0 | 0 |
| **graph_only** | 0 | 5 | 0 |
| **graph+vector** | 0 | 0 | 2 |

## 3. Sample-by-Sample Details

| # | Query | Expected | Predicted | Intent | Match |
|---|---|---|---|---|:---:|
| 1 | Which concepts depend on virtual memory according to th... | `graph_only` | `graph_only` | `factual_qa` | PASS |
| 2 | What topics are related to concurrency in the roadmap f... | `graph_only` | `graph_only` | `factual_qa` | PASS |
| 3 | What is the path from the process abstraction to the re... | `graph_only` | `graph_only` | `factual_qa` | PASS |
| 4 | What was taught before the discussion of deadlock in th... | `graph_only` | `graph_only` | `factual_qa` | PASS |
| 5 | How is the thread concept connected to the kernel and t... | `graph_only` | `graph_only` | `factual_qa` | PASS |
| 6 | Explain the relationship between processes and threads ... | `graph+vector` | `graph+vector` | `conceptual` | PASS |
| 7 | Explain how virtual memory and page tables work togethe... | `vector_only` | `vector_only` | `factual_qa` | PASS |
| 8 | Explain what the kernel provides and how file systems a... | `graph+vector` | `graph+vector` | `conceptual` | PASS |
| 9 | What was said about CPU scheduling later in the term?... | `vector_only` | `vector_only` | `definition` | PASS |
| 10 | How does the lecture describe protecting programs from ... | `vector_only` | `vector_only` | `factual_qa` | PASS |
| 11 | What was said about why many layers of abstraction are ... | `vector_only` | `vector_only` | `reasoning` | PASS |
| 12 | Describe the devices the lecture says the OS must tie t... | `vector_only` | `vector_only` | `factual_qa` | PASS |

## 4. Benchmark Scope & Statistical Limitations

- **Sample size:** N = 12 questions.
- **Resolution:** 1 sample = 8.33% of total accuracy.
- **Coverage:** This dataset explicitly balances structural/graph queries (prerequisites, connections),
  pure semantic explanation queries (vector_only), and composite relational+explanatory queries (graph+vector).
- **Note on Headline Metrics:** The 50-QA benchmark (`cs162_lecture1_qa_50.json`) contains only `vector_only`
  labels because it targets multimodal chunk retrieval. True multi-route classification performance
  must be cited from this balanced 12-query routing benchmark, not the 50-QA benchmark.
