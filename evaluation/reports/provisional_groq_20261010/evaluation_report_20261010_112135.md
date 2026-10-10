# LectureMind Evaluation Report
**Generated:** 20261010_112135
**Samples:** 12 (12 successful, 0 failed)

## Aggregated Metrics

| Metric | Mean | p95 | Min | Max | n |
|---|---|---|---|---|---|
| answer_f1 | 0.2898 | 0.5714 | 0.0476 | 0.5714 | 12 |
| answer_length | 201.3333 | 320.0000 | 39.0000 | 320.0000 | 12 |
| avg_cross_encoder_score | 0.0937 | 0.4411 | 0.0001 | 0.4411 | 12 |
| chunk_coverage | 0.1667 | 0.2667 | 0.0667 | 0.2667 | 12 |
| citation_count | 2.5000 | 4.0000 | 1.0000 | 4.0000 | 12 |
| citation_coverage | 0.7500 | 1.0000 | 0.0000 | 1.0000 | 12 |
| citation_provenance_precision | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 12 |
| context_length | 3144.4167 | 3891.0000 | 1521.0000 | 3891.0000 | 12 |
| generation_latency | 7.7944 | 15.3856 | 0.6465 | 15.3856 | 12 |
| hit_at_5 | 0.9167 | 1.0000 | 0.0000 | 1.0000 | 12 |
| keyword_recall | 0.6034 | 1.0000 | 0.0000 | 1.0000 | 12 |
| lexical_content_jaccard | 0.1914 | 0.5714 | 0.0000 | 0.5714 | 12 |
| mrr | 0.6579 | 1.0000 | 0.1111 | 1.0000 | 12 |
| mrr_at_5 | 0.6486 | 1.0000 | 0.0000 | 1.0000 | 12 |
| ndcg_at_5 | 0.7149 | 1.0000 | 0.0000 | 1.0000 | 12 |
| planner_latency | 0.0007 | 0.0038 | 0.0001 | 0.0038 | 12 |
| precision_at_5 | 0.1833 | 0.2000 | 0.0000 | 0.2000 | 12 |
| r_precision | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 12 |
| ranking_quality | 0.7629 | 1.0000 | 0.0714 | 1.0000 | 12 |
| recall_at_5 | 0.9167 | 1.0000 | 0.0000 | 1.0000 | 12 |
| reranker_latency | 5.0506 | 7.3138 | 2.7465 | 7.3138 | 12 |
| retrieval_latency | 0.1649 | 0.3983 | 0.1074 | 0.3983 | 12 |
| routing_accuracy | 0.8333 | 1.0000 | 0.0000 | 1.0000 | 12 |
| top1_cross_encoder_score | 0.5879 | 0.9978 | 0.0002 | 0.9978 | 12 |
| top3_cross_encoder_score | 0.3313 | 0.9631 | 0.0001 | 0.9631 | 12 |
| total_latency | 13.0106 | 19.1700 | 4.5958 | 19.1700 | 12 |
| visual_routing_accuracy | 0.7500 | 1.0000 | 0.0000 | 1.0000 | 12 |

## By Question Type

| Group | Metric | Mean |
|---|---|---|
| factual | answer_f1 | 0.1541 |
| factual | answer_length | 170.5000 |
| factual | avg_cross_encoder_score | 0.0386 |
| factual | chunk_coverage | 0.2000 |
| factual | citation_count | 3.0000 |
| factual | citation_coverage | 1.0000 |
| factual | citation_provenance_precision | 1.0000 |
| factual | context_length | 3378.5000 |
| factual | generation_latency | 1.1321 |
| factual | hit_at_5 | 1.0000 |
| factual | keyword_recall | 0.6667 |
| factual | lexical_content_jaccard | 0.1576 |
| factual | mrr | 0.7500 |
| factual | mrr_at_5 | 0.7500 |
| factual | ndcg_at_5 | 0.8155 |
| factual | planner_latency | 0.0020 |
| factual | precision_at_5 | 0.2000 |
| factual | r_precision | 0.5000 |
| factual | ranking_quality | 0.6667 |
| factual | recall_at_5 | 1.0000 |
| factual | reranker_latency | 4.0397 |
| factual | retrieval_latency | 0.1123 |
| factual | routing_accuracy | 1.0000 |
| factual | top1_cross_encoder_score | 0.5002 |
| factual | top3_cross_encoder_score | 0.1917 |
| factual | total_latency | 5.2860 |
| factual | visual_routing_accuracy | 1.0000 |
| visual | answer_f1 | 0.3170 |
| visual | answer_length | 207.5000 |
| visual | avg_cross_encoder_score | 0.1048 |
| visual | chunk_coverage | 0.1600 |
| visual | citation_count | 2.4000 |
| visual | citation_coverage | 0.7000 |
| visual | citation_provenance_precision | 1.0000 |
| visual | context_length | 3097.6000 |
| visual | generation_latency | 9.1269 |
| visual | hit_at_5 | 0.9000 |
| visual | keyword_recall | 0.5907 |
| visual | lexical_content_jaccard | 0.1981 |
| visual | mrr | 0.6394 |
| visual | mrr_at_5 | 0.6283 |
| visual | ndcg_at_5 | 0.6948 |
| visual | planner_latency | 0.0004 |
| visual | precision_at_5 | 0.1800 |
| visual | r_precision | 0.5000 |
| visual | ranking_quality | 0.7821 |
| visual | recall_at_5 | 0.9000 |
| visual | reranker_latency | 5.2528 |
| visual | retrieval_latency | 0.1754 |
| visual | routing_accuracy | 0.8000 |
| visual | top1_cross_encoder_score | 0.6055 |
| visual | top3_cross_encoder_score | 0.3593 |
| visual | total_latency | 14.5555 |
| visual | visual_routing_accuracy | 0.7000 |

## By Difficulty

| Group | Metric | Mean |
|---|---|---|
| easy | answer_f1 | 0.2859 |
| easy | answer_length | 190.5455 |
| easy | avg_cross_encoder_score | 0.0912 |
| easy | chunk_coverage | 0.1758 |
| easy | citation_count | 2.6364 |
| easy | citation_coverage | 0.8182 |
| easy | citation_provenance_precision | 1.0000 |
| easy | context_length | 3292.0000 |
| easy | generation_latency | 7.7871 |
| easy | hit_at_5 | 0.9091 |
| easy | keyword_recall | 0.6355 |
| easy | lexical_content_jaccard | 0.1970 |
| easy | mrr | 0.6268 |
| easy | mrr_at_5 | 0.6167 |
| easy | ndcg_at_5 | 0.6890 |
| easy | planner_latency | 0.0007 |
| easy | precision_at_5 | 0.1818 |
| easy | r_precision | 0.4545 |
| easy | ranking_quality | 0.7868 |
| easy | recall_at_5 | 0.9091 |
| easy | reranker_latency | 4.9991 |
| easy | retrieval_latency | 0.1437 |
| easy | routing_accuracy | 0.9091 |
| easy | top1_cross_encoder_score | 0.5536 |
| easy | top3_cross_encoder_score | 0.3077 |
| easy | total_latency | 12.9306 |
| easy | visual_routing_accuracy | 0.7273 |
| medium | answer_f1 | 0.3333 |
| medium | answer_length | 320.0000 |
| medium | avg_cross_encoder_score | 0.1217 |
| medium | chunk_coverage | 0.0667 |
| medium | citation_count | 1.0000 |
| medium | citation_coverage | 0.0000 |
| medium | citation_provenance_precision | 1.0000 |
| medium | context_length | 1521.0000 |
| medium | generation_latency | 7.8748 |
| medium | hit_at_5 | 1.0000 |
| medium | keyword_recall | 0.2500 |
| medium | lexical_content_jaccard | 0.1290 |
| medium | mrr | 1.0000 |
| medium | mrr_at_5 | 1.0000 |
| medium | ndcg_at_5 | 1.0000 |
| medium | planner_latency | 0.0004 |
| medium | precision_at_5 | 0.2000 |
| medium | r_precision | 1.0000 |
| medium | ranking_quality | 0.5000 |
| medium | recall_at_5 | 1.0000 |
| medium | reranker_latency | 5.6176 |
| medium | retrieval_latency | 0.3983 |
| medium | routing_accuracy | 0.0000 |
| medium | top1_cross_encoder_score | 0.9650 |
| medium | top3_cross_encoder_score | 0.5910 |
| medium | total_latency | 13.8911 |
| medium | visual_routing_accuracy | 1.0000 |

## Per-Sample Results

| # | Type | Difficulty | Query | Route | MRR@5 | Recall@5 | Answer F1 | Citation | Total Lat (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | factual | easy | What is the instructor's name mentioned in the fir | vector_only | 0.500 | 1.000 | 0.118 | 1.000 | 5.98 |
| 2 | factual | easy | Who is the instructor for MIT 6.S191? | vector_only | 1.000 | 1.000 | 0.190 | 1.000 | 4.60 |
| 3 | visual | easy | Which brand of microphone is visible in the podcas | vector_only | 0.000 | 0.000 | 0.133 | 0.000 | 7.08 |
| 4 | visual | easy | According to the slide, which of the following is  | vector_only | 1.000 | 1.000 | 0.318 | 1.000 | 8.46 |
| 5 | visual | easy | Which hardware component is connected to the cache | vector_only | 0.250 | 1.000 | 0.133 | 1.000 | 6.70 |
| 6 | visual | easy | What role does the 'Referee' play in the operating | vector_only | 0.200 | 1.000 | 0.453 | 0.000 | 19.15 |
| 7 | visual | easy | According to the slide, what is the purpose of com | vector_only | 0.333 | 1.000 | 0.426 | 1.000 | 18.96 |
| 8 | visual | easy | What device was used to run the live demonstration | vector_only | 1.000 | 1.000 | 0.048 | 1.000 | 19.17 |
| 9 | visual | easy | Which companies and institutions are listed as spo | vector_only | 1.000 | 1.000 | 0.571 | 1.000 | 17.26 |
| 10 | visual | easy | In the 2D embedding plot described, what do the X‑ | vector_only | 0.500 | 1.000 | 0.354 | 1.000 | 17.96 |
| 11 | visual | easy | In the educational slide, which Q vector is used w | vector_only | 1.000 | 1.000 | 0.400 | 1.000 | 16.92 |
| 12 | visual | medium | What feature is highlighted in the educational sli | vector_only | 1.000 | 1.000 | 0.333 | 0.000 | 13.89 |
