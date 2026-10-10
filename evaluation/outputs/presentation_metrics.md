# LectureMIND — Current Interview Metrics

This is the current three-lecture smoke evaluation, not a production benchmark.
It contains 45 questions (15 per lecture), uses the active local BGE reranker,
Qdrant, Neo4j, and the configured Groq backend, and is provisional because the
labels are not yet human-reviewed.

Source: `evaluation/reports/smoke_groq_20261010_fixed/`

## Current headline metrics

| Metric | Result | Denominator / qualification |
|---|---:|---|
| Hit@5 | **0.8605** | 43 answerable questions |
| Recall@5 | **0.8488** | 43 answerable questions |
| MRR@5 | **0.7333** | 43 answerable questions |
| nDCG@5 | **0.7595** | 43 answerable questions |
| Precision@5 | **0.1721** | 43 answerable questions |
| Reranker ranking quality | **0.7644** | 43 answerable questions |
| Citation coverage | **0.7674** | 43 answerable questions |
| Answer F1 | **0.2667** | 45 questions; lexical metric |
| Keyword recall | **0.7540** | 45 questions |
| Routing accuracy | **0.8889** | 45 questions |
| Visual routing accuracy | **0.9111** | 45 questions |
| Mean total latency | **17.32 s** | 45 questions |

Citation provenance precision is 1.000, but this is a structural check that cited
IDs came from the supplied context; it is not claim-level factual faithfulness.
The two unanswerable samples are excluded from evidence-based retrieval metrics.

## Interpretation

- Retrieval is stronger than answer generation: Recall@5 is 84.88%, while Answer F1 is 26.67%.
- Graph questions are the weakest slice and need further improvement.
- The evaluation bug that treated empty unanswerable evidence as perfect retrieval was corrected.
- The evaluation is suitable for a prototype demonstration, not a production-quality claim.

## Defensible interview wording

> “I evaluated the live LectureMIND pipeline on 45 questions across three lectures. The
> corrected smoke run achieved 84.88% Recall@5, 73.33% MRR@5, and 76.44% reranker
> ranking quality. Answer F1 was 26.67%, showing that retrieval quality is ahead of
> answer quality. The benchmark is provisional because the labels still need human review.”

## Archived results

The older 50-question CS162 artifacts remain under `evaluation/outputs/` for forensic
traceability. They are historical and must not be presented as current results.
