# LectureMind Evaluation Report
**Generated:** 20261114_154651
**Samples:** 45 (45 successful, 0 failed)

## Aggregated Metrics

| Metric | Mean | p95 | Min | Max | n |
|---|---|---|---|---|---|
| answer_f1 | 0.2530 | 0.5000 | 0.0000 | 0.6538 | 45 |
| answer_length | 850.1111 | 3199.0000 | 39.0000 | 3753.0000 | 45 |
| avg_cross_encoder_score | 0.1057 | 0.3386 | 0.0001 | 0.4623 | 43 |
| chunk_coverage | 0.2148 | 0.3333 | 0.0667 | 0.3333 | 45 |
| citation_count | 3.2222 | 5.0000 | 1.0000 | 5.0000 | 45 |
| citation_coverage | 0.7674 | 1.0000 | 0.0000 | 1.0000 | 43 |
| citation_provenance_precision | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 45 |
| context_length | 3271.9556 | 3891.0000 | 1671.0000 | 3973.0000 | 45 |
| context_recall | 0.7674 | 1.0000 | 0.0000 | 1.0000 | 43 |
| generation_latency | 16.0119 | 21.5069 | 0.7449 | 22.8325 | 45 |
| hit_at_5 | 0.8605 | 1.0000 | 0.0000 | 1.0000 | 43 |
| keyword_recall | 0.7733 | 1.0000 | 0.0000 | 1.0000 | 45 |
| lexical_content_jaccard | 0.1904 | 0.4286 | 0.0000 | 0.4706 | 45 |
| mrr | 0.7438 | 1.0000 | 0.0000 | 1.0000 | 43 |
| mrr_at_5 | 0.7333 | 1.0000 | 0.0000 | 1.0000 | 43 |
| ndcg_at_5 | 0.7595 | 1.0000 | 0.0000 | 1.0000 | 43 |
| planner_latency | 0.0005 | 0.0008 | 0.0001 | 0.0039 | 45 |
| post_rerank_hit_at_5 | 0.8837 | 1.0000 | 0.0000 | 1.0000 | 43 |
| post_rerank_mrr_at_5 | 0.7593 | 1.0000 | 0.0000 | 1.0000 | 43 |
| post_rerank_ndcg_at_5 | 0.7854 | 1.0000 | 0.0000 | 1.0000 | 43 |
| post_rerank_recall_at_5 | 0.8721 | 1.0000 | 0.0000 | 1.0000 | 43 |
| precision_at_5 | 0.1721 | 0.2000 | 0.0000 | 0.2000 | 43 |
| r_precision | 0.6628 | 1.0000 | 0.0000 | 1.0000 | 43 |
| ranking_quality | 0.7644 | 1.0000 | 0.0000 | 1.0000 | 43 |
| recall_at_5 | 0.8488 | 1.0000 | 0.0000 | 1.0000 | 43 |
| refusal_correct | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 2 |
| reranker_latency | 5.6083 | 10.5930 | 2.3508 | 12.8887 | 45 |
| retrieval_latency | 0.2002 | 0.5062 | 0.1051 | 0.6548 | 45 |
| routing_accuracy | 0.9333 | 1.0000 | 0.0000 | 1.0000 | 45 |
| top1_cross_encoder_score | 0.6066 | 0.9991 | 0.0002 | 0.9995 | 43 |
| top3_cross_encoder_score | 0.3726 | 0.9557 | 0.0001 | 0.9759 | 43 |
| total_latency | 21.8209 | 24.6350 | 4.8740 | 25.3164 | 45 |
| unanswerable_retrieval_empty | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2 |
| visual_routing_accuracy | 0.9111 | 1.0000 | 0.0000 | 1.0000 | 45 |

## By Question Type

| Group | Metric | Mean |
|---|---|---|
| conceptual | answer_f1 | 0.2692 |
| conceptual | answer_length | 1570.8333 |
| conceptual | avg_cross_encoder_score | 0.1140 |
| conceptual | chunk_coverage | 0.2444 |
| conceptual | citation_count | 3.6667 |
| conceptual | citation_coverage | 0.9167 |
| conceptual | citation_provenance_precision | 1.0000 |
| conceptual | context_length | 3238.0000 |
| conceptual | context_recall | 0.9167 |
| conceptual | generation_latency | 16.7538 |
| conceptual | hit_at_5 | 0.9167 |
| conceptual | keyword_recall | 0.9028 |
| conceptual | lexical_content_jaccard | 0.2326 |
| conceptual | mrr | 0.9226 |
| conceptual | mrr_at_5 | 0.9167 |
| conceptual | ndcg_at_5 | 0.9167 |
| conceptual | planner_latency | 0.0004 |
| conceptual | post_rerank_hit_at_5 | 0.9167 |
| conceptual | post_rerank_mrr_at_5 | 0.8194 |
| conceptual | post_rerank_ndcg_at_5 | 0.8442 |
| conceptual | post_rerank_recall_at_5 | 0.9167 |
| conceptual | precision_at_5 | 0.1833 |
| conceptual | r_precision | 0.9167 |
| conceptual | ranking_quality | 0.8259 |
| conceptual | recall_at_5 | 0.9167 |
| conceptual | reranker_latency | 4.6173 |
| conceptual | retrieval_latency | 0.1525 |
| conceptual | routing_accuracy | 0.9167 |
| conceptual | top1_cross_encoder_score | 0.6109 |
| conceptual | top3_cross_encoder_score | 0.4067 |
| conceptual | total_latency | 21.5240 |
| conceptual | visual_routing_accuracy | 1.0000 |
| definition | answer_f1 | 0.4324 |
| definition | answer_length | 347.0000 |
| definition | avg_cross_encoder_score | 0.0067 |
| definition | chunk_coverage | 0.2667 |
| definition | citation_count | 4.0000 |
| definition | citation_coverage | 1.0000 |
| definition | citation_provenance_precision | 1.0000 |
| definition | context_length | 3242.0000 |
| definition | context_recall | 1.0000 |
| definition | generation_latency | 17.1489 |
| definition | hit_at_5 | 1.0000 |
| definition | keyword_recall | 1.0000 |
| definition | lexical_content_jaccard | 0.2069 |
| definition | mrr | 1.0000 |
| definition | mrr_at_5 | 1.0000 |
| definition | ndcg_at_5 | 1.0000 |
| definition | planner_latency | 0.0001 |
| definition | post_rerank_hit_at_5 | 1.0000 |
| definition | post_rerank_mrr_at_5 | 1.0000 |
| definition | post_rerank_ndcg_at_5 | 1.0000 |
| definition | post_rerank_recall_at_5 | 1.0000 |
| definition | precision_at_5 | 0.2000 |
| definition | r_precision | 1.0000 |
| definition | ranking_quality | 1.0000 |
| definition | recall_at_5 | 1.0000 |
| definition | reranker_latency | 5.3729 |
| definition | retrieval_latency | 0.1331 |
| definition | routing_accuracy | 1.0000 |
| definition | top1_cross_encoder_score | 0.0355 |
| definition | top3_cross_encoder_score | 0.0269 |
| definition | total_latency | 22.6550 |
| definition | visual_routing_accuracy | 1.0000 |
| factual | answer_f1 | 0.2117 |
| factual | answer_length | 948.5882 |
| factual | avg_cross_encoder_score | 0.1201 |
| factual | chunk_coverage | 0.2667 |
| factual | citation_count | 4.0000 |
| factual | citation_coverage | 0.8235 |
| factual | citation_provenance_precision | 1.0000 |
| factual | context_length | 3431.7647 |
| factual | context_recall | 0.8235 |
| factual | generation_latency | 16.8466 |
| factual | hit_at_5 | 0.8824 |
| factual | keyword_recall | 0.8158 |
| factual | lexical_content_jaccard | 0.1870 |
| factual | mrr | 0.7500 |
| factual | mrr_at_5 | 0.7500 |
| factual | ndcg_at_5 | 0.7837 |
| factual | planner_latency | 0.0006 |
| factual | post_rerank_hit_at_5 | 0.8824 |
| factual | post_rerank_mrr_at_5 | 0.7961 |
| factual | post_rerank_ndcg_at_5 | 0.8169 |
| factual | post_rerank_recall_at_5 | 0.8824 |
| factual | precision_at_5 | 0.1765 |
| factual | r_precision | 0.6471 |
| factual | ranking_quality | 0.7961 |
| factual | recall_at_5 | 0.8824 |
| factual | reranker_latency | 4.4649 |
| factual | retrieval_latency | 0.1496 |
| factual | routing_accuracy | 1.0000 |
| factual | top1_cross_encoder_score | 0.7183 |
| factual | top3_cross_encoder_score | 0.4241 |
| factual | total_latency | 21.4617 |
| factual | visual_routing_accuracy | 0.9412 |
| graph | answer_f1 | 0.4975 |
| graph | answer_length | 238.0000 |
| graph | avg_cross_encoder_score | 0.0270 |
| graph | chunk_coverage | 0.0667 |
| graph | citation_count | 1.0000 |
| graph | citation_coverage | 0.0000 |
| graph | citation_provenance_precision | 1.0000 |
| graph | context_length | 2926.6667 |
| graph | context_recall | 0.0000 |
| graph | generation_latency | 11.0745 |
| graph | hit_at_5 | 0.3333 |
| graph | keyword_recall | 1.0000 |
| graph | lexical_content_jaccard | 0.2749 |
| graph | mrr | 0.2560 |
| graph | mrr_at_5 | 0.1667 |
| graph | ndcg_at_5 | 0.1290 |
| graph | planner_latency | 0.0004 |
| graph | post_rerank_hit_at_5 | 0.6667 |
| graph | post_rerank_mrr_at_5 | 0.1778 |
| graph | post_rerank_ndcg_at_5 | 0.2311 |
| graph | post_rerank_recall_at_5 | 0.5000 |
| graph | precision_at_5 | 0.0667 |
| graph | r_precision | 0.1667 |
| graph | ranking_quality | 0.2016 |
| graph | recall_at_5 | 0.1667 |
| graph | reranker_latency | 10.4930 |
| graph | retrieval_latency | 0.5797 |
| graph | routing_accuracy | 1.0000 |
| graph | top1_cross_encoder_score | 0.1501 |
| graph | top3_cross_encoder_score | 0.1044 |
| graph | total_latency | 22.1475 |
| graph | visual_routing_accuracy | 1.0000 |
| unanswerable | answer_f1 | 0.0991 |
| unanswerable | answer_length | 39.0000 |
| unanswerable | chunk_coverage | 0.0667 |
| unanswerable | citation_count | 1.0000 |
| unanswerable | citation_provenance_precision | 1.0000 |
| unanswerable | context_length | 2950.5000 |
| unanswerable | generation_latency | 12.7997 |
| unanswerable | keyword_recall | 0.0000 |
| unanswerable | lexical_content_jaccard | 0.0426 |
| unanswerable | planner_latency | 0.0005 |
| unanswerable | refusal_correct | 1.0000 |
| unanswerable | reranker_latency | 8.9684 |
| unanswerable | retrieval_latency | 0.3760 |
| unanswerable | routing_accuracy | 1.0000 |
| unanswerable | total_latency | 22.1446 |
| unanswerable | unanswerable_retrieval_empty | 0.0000 |
| unanswerable | visual_routing_accuracy | 1.0000 |
| visual | answer_f1 | 0.2431 |
| visual | answer_length | 214.0000 |
| visual | avg_cross_encoder_score | 0.1048 |
| visual | chunk_coverage | 0.1600 |
| visual | citation_count | 2.4000 |
| visual | citation_coverage | 0.7000 |
| visual | citation_provenance_precision | 1.0000 |
| visual | context_length | 3211.9000 |
| visual | context_recall | 0.7000 |
| visual | generation_latency | 15.7128 |
| visual | hit_at_5 | 0.9000 |
| visual | keyword_recall | 0.6098 |
| visual | lexical_content_jaccard | 0.1482 |
| visual | mrr | 0.6394 |
| visual | mrr_at_5 | 0.6283 |
| visual | ndcg_at_5 | 0.6948 |
| visual | planner_latency | 0.0005 |
| visual | post_rerank_hit_at_5 | 0.9000 |
| visual | post_rerank_mrr_at_5 | 0.7750 |
| visual | post_rerank_ndcg_at_5 | 0.8062 |
| visual | post_rerank_recall_at_5 | 0.9000 |
| visual | precision_at_5 | 0.1800 |
| visual | r_precision | 0.5000 |
| visual | ranking_quality | 0.7821 |
| visual | recall_at_5 | 0.9000 |
| visual | reranker_latency | 6.6274 |
| visual | retrieval_latency | 0.2011 |
| visual | routing_accuracy | 0.8000 |
| visual | top1_cross_encoder_score | 0.6055 |
| visual | top3_cross_encoder_score | 0.3593 |
| visual | total_latency | 22.5418 |
| visual | visual_routing_accuracy | 0.7000 |

## By Difficulty

| Group | Metric | Mean |
|---|---|---|
| easy | answer_f1 | 0.2438 |
| easy | answer_length | 672.7500 |
| easy | avg_cross_encoder_score | 0.1071 |
| easy | chunk_coverage | 0.2381 |
| easy | citation_count | 3.5714 |
| easy | citation_coverage | 0.8214 |
| easy | citation_provenance_precision | 1.0000 |
| easy | context_length | 3390.7500 |
| easy | context_recall | 0.8214 |
| easy | generation_latency | 16.6092 |
| easy | hit_at_5 | 0.8929 |
| easy | keyword_recall | 0.7845 |
| easy | lexical_content_jaccard | 0.1884 |
| easy | mrr | 0.7194 |
| easy | mrr_at_5 | 0.7155 |
| easy | ndcg_at_5 | 0.7597 |
| easy | planner_latency | 0.0005 |
| easy | post_rerank_hit_at_5 | 0.8929 |
| easy | post_rerank_mrr_at_5 | 0.8137 |
| easy | post_rerank_ndcg_at_5 | 0.8328 |
| easy | post_rerank_recall_at_5 | 0.8929 |
| easy | precision_at_5 | 0.1786 |
| easy | r_precision | 0.6071 |
| easy | ranking_quality | 0.8162 |
| easy | recall_at_5 | 0.8929 |
| easy | reranker_latency | 5.1888 |
| easy | retrieval_latency | 0.1666 |
| easy | routing_accuracy | 0.9643 |
| easy | top1_cross_encoder_score | 0.6312 |
| easy | top3_cross_encoder_score | 0.3701 |
| easy | total_latency | 21.9651 |
| easy | visual_routing_accuracy | 0.8571 |
| medium | answer_f1 | 0.2681 |
| medium | answer_length | 1142.2353 |
| medium | avg_cross_encoder_score | 0.1030 |
| medium | chunk_coverage | 0.1765 |
| medium | citation_count | 2.6471 |
| medium | citation_coverage | 0.6667 |
| medium | citation_provenance_precision | 1.0000 |
| medium | context_length | 3076.2941 |
| medium | context_recall | 0.6667 |
| medium | generation_latency | 15.0281 |
| medium | hit_at_5 | 0.8000 |
| medium | keyword_recall | 0.7549 |
| medium | lexical_content_jaccard | 0.1937 |
| medium | mrr | 0.7893 |
| medium | mrr_at_5 | 0.7667 |
| medium | ndcg_at_5 | 0.7591 |
| medium | planner_latency | 0.0004 |
| medium | post_rerank_hit_at_5 | 0.8667 |
| medium | post_rerank_mrr_at_5 | 0.6578 |
| medium | post_rerank_ndcg_at_5 | 0.6970 |
| medium | post_rerank_recall_at_5 | 0.8333 |
| medium | precision_at_5 | 0.1600 |
| medium | r_precision | 0.7667 |
| medium | ranking_quality | 0.6677 |
| medium | recall_at_5 | 0.7667 |
| medium | refusal_correct | 1.0000 |
| medium | reranker_latency | 6.2994 |
| medium | retrieval_latency | 0.2555 |
| medium | routing_accuracy | 0.8824 |
| medium | top1_cross_encoder_score | 0.5605 |
| medium | top3_cross_encoder_score | 0.3774 |
| medium | total_latency | 21.5834 |
| medium | unanswerable_retrieval_empty | 0.0000 |
| medium | visual_routing_accuracy | 1.0000 |

## Per-Sample Results

| # | Type | Difficulty | Query | Route | MRR@5 | Recall@5 | Answer F1 | Citation | Total Lat (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | factual | easy | What is the instructor's name mentioned in the fir | vector_only | 0.500 | 1.000 | 0.167 | 1.000 | 6.71 |
| 2 | factual | easy | How many devices could the original ARPANET suppor | vector_only | 1.000 | 1.000 | 0.333 | 1.000 | 4.87 |
| 3 | conceptual | medium | What topics related to security will be covered in | vector_only | 1.000 | 1.000 | 0.491 | 1.000 | 6.89 |
| 4 | conceptual | medium | What are the main qualities highlighted in the lec | vector_only | 1.000 | 1.000 | 0.478 | 1.000 | 18.07 |
| 5 | factual | easy | Will the lecture videos be automatically captioned | vector_only | 1.000 | 1.000 | 0.500 | 1.000 | 23.66 |
| 6 | conceptual | medium | Why are multiple layers of abstraction necessary i | vector_only | 1.000 | 1.000 | 0.049 | 1.000 | 25.30 |
| 7 | visual | easy | According to the slide, which of the following is  | vector_only | 1.000 | 1.000 | 0.281 | 1.000 | 23.05 |
| 8 | visual | easy | Which hardware component is connected to the cache | vector_only | 0.250 | 1.000 | 0.143 | 1.000 | 23.33 |
| 9 | visual | easy | What role does the 'Referee' play in the operating | vector_only | 0.200 | 1.000 | 0.367 | 0.000 | 24.07 |
| 10 | visual | easy | According to the slide, what is the purpose of com | vector_only | 0.333 | 1.000 | 0.426 | 1.000 | 24.37 |
| 11 | graph | medium | What is the prerequisite relationship between Oper | graph_only | 0.000 | 0.000 | 0.484 | 0.000 | 22.74 |
| 12 | graph | medium | How is Process Abstraction related to Process in t | graph_only | 0.000 | 0.000 | 0.654 | 0.000 | 22.47 |
| 13 | graph | medium | Which subsystem does Kernel connect to via USED_BY | graph_only | 0.500 | 0.500 | 0.355 | 0.000 | 21.23 |
| 14 | unanswerable | medium | What direct prerequisite relationship exists betwe | graph_only | n/a | n/a | 0.093 | n/a | 21.67 |
| 15 | unanswerable | medium | How is Bell's Law a prerequisite for the implement | graph_only | n/a | n/a | 0.105 | n/a | 22.62 |
| 16 | factual | easy | Who is the instructor for MIT 6.S191? | vector_only | 1.000 | 1.000 | 0.154 | 1.000 | 23.04 |
| 17 | factual | easy | What are the infrastructure and cost requirements  | vector_only | 1.000 | 1.000 | 0.426 | 1.000 | 24.53 |
| 18 | conceptual | medium | According to the lecture, what is the difference b | vector_only | 1.000 | 1.000 | 0.126 | 1.000 | 23.24 |
| 19 | factual | easy | Which programming frameworks will be offered for t | vector_only | 1.000 | 1.000 | 0.154 | 1.000 | 22.68 |
| 20 | factual | easy | What are the three main factors that have enabled  | vector_only | 0.000 | 0.000 | 0.111 | 0.000 | 21.43 |
| 21 | conceptual | easy | What role does the bias term play in a neuron? | vector_only | 1.000 | 1.000 | 0.438 | 1.000 | 24.63 |
| 22 | conceptual | medium | Why are activation functions necessary in neural n | vector_only | 1.000 | 1.000 | 0.065 | 1.000 | 23.11 |
| 23 | conceptual | medium | What is the sign of the neuron's output for the po | vector_only | 1.000 | 1.000 | 0.600 | 1.000 | 23.02 |
| 24 | factual | easy | What are the three steps involved in the operation | vector_only | 1.000 | 1.000 | 0.342 | 1.000 | 22.45 |
| 25 | factual | easy | How do TensorFlow and PyTorch allow users to creat | vector_only | 1.000 | 1.000 | 0.225 | 1.000 | 23.03 |
| 26 | conceptual | medium | Why are fully connected layers stacked with nonlin | vector_only | 1.000 | 1.000 | 0.081 | 1.000 | 23.14 |
| 27 | factual | easy | What probability does the neural network predict f | vector_only | 1.000 | 1.000 | 0.476 | 1.000 | 24.17 |
| 28 | factual | easy | Why do classification networks output values betwe | vector_only | 0.500 | 1.000 | 0.088 | 1.000 | 24.08 |
| 29 | visual | easy | What device was used to run the live demonstration | vector_only | 1.000 | 1.000 | 0.000 | 1.000 | 22.63 |
| 30 | visual | easy | Which companies and institutions are listed as spo | vector_only | 1.000 | 1.000 | 0.370 | 1.000 | 22.97 |
| 31 | conceptual | medium | What are the three different contexts in which the | vector_only | 1.000 | 1.000 | 0.341 | 1.000 | 20.04 |
| 32 | conceptual | medium | Why is a static word embedding insufficient for re | vector_only | 1.000 | 1.000 | 0.119 | 1.000 | 23.92 |
| 33 | conceptual | medium | How does self-attention modify the embedding of a  | vector_only | 1.000 | 1.000 | 0.373 | 1.000 | 23.16 |
| 34 | definition | easy | According to the speaker, what determines the mean | vector_only | 1.000 | 1.000 | 0.432 | 1.000 | 22.65 |
| 35 | factual | easy | What effect does adding phone embeddings have on t | vector_only | 1.000 | 1.000 | 0.095 | 1.000 | 24.18 |
| 36 | conceptual | medium | Why do the coefficients used for combining word em | vector_only | 0.000 | 0.000 | 0.071 | 0.000 | 23.75 |
| 37 | factual | easy | How is the similarity between two words determined | vector_only | 1.000 | 1.000 | 0.057 | 1.000 | 23.63 |
| 38 | factual | easy | Which function is used to convert similarity score | vector_only | 0.500 | 1.000 | 0.140 | 1.000 | 23.45 |
| 39 | factual | easy | What is one limitation of using only fixed word em | vector_only | 1.000 | 1.000 | 0.277 | 1.000 | 23.37 |
| 40 | factual | easy | Why does the speaker want to introduce learnable p | vector_only | 0.250 | 1.000 | 0.049 | 0.000 | 25.32 |
| 41 | visual | easy | Which brand of microphone is visible in the podcas | vector_only | 0.000 | 0.000 | 0.133 | 0.000 | 17.14 |
| 42 | visual | easy | In the 2D embedding plot described, what do the X‑ | vector_only | 0.500 | 1.000 | 0.309 | 1.000 | 23.80 |
| 43 | factual | easy | Which function is used to normalize similarity sco | vector_only | 0.000 | 0.000 | 0.006 | 0.000 | 24.25 |
| 44 | visual | easy | In the educational slide, which Q vector is used w | vector_only | 1.000 | 1.000 | 0.327 | 1.000 | 21.53 |
| 45 | visual | medium | What feature is highlighted in the educational sli | vector_only | 1.000 | 1.000 | 0.074 | 0.000 | 22.53 |
