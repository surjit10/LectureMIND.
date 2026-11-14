# LectureMind Evaluation Report
**Generated:** 20261114_171933
**Samples:** 45 (45 successful, 0 failed)

## Aggregated Metrics

| Metric | Mean | p95 | Min | Max | n |
|---|---|---|---|---|---|
| answer_f1 | 0.2559 | 0.5538 | 0.0000 | 0.5714 | 45 |
| answer_length | 888.3778 | 3197.0000 | 39.0000 | 3883.0000 | 45 |
| avg_cross_encoder_score | 0.1057 | 0.3386 | 0.0001 | 0.4623 | 43 |
| chunk_coverage | 0.3674 | 0.4667 | 0.1333 | 0.4667 | 45 |
| citation_count | 5.5111 | 7.0000 | 2.0000 | 7.0000 | 45 |
| citation_coverage | 0.8430 | 1.0000 | 0.0000 | 1.0000 | 43 |
| citation_provenance_precision | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 45 |
| context_length | 5547.6444 | 5965.0000 | 3891.0000 | 5989.0000 | 45 |
| context_recall | 0.8430 | 1.0000 | 0.0000 | 1.0000 | 43 |
| generation_latency | 22.8201 | 27.5315 | 0.7242 | 36.7001 | 45 |
| hit_at_5 | 0.8837 | 1.0000 | 0.0000 | 1.0000 | 43 |
| keyword_recall | 0.7857 | 1.0000 | 0.0000 | 1.0000 | 45 |
| lexical_content_jaccard | 0.1995 | 0.5000 | 0.0000 | 0.6316 | 45 |
| mrr | 0.8147 | 1.0000 | 0.0000 | 1.0000 | 43 |
| mrr_at_5 | 0.8043 | 1.0000 | 0.0000 | 1.0000 | 43 |
| ndcg_at_5 | 0.7877 | 1.0000 | 0.0000 | 1.0000 | 43 |
| planner_latency | 0.0005 | 0.0009 | 0.0001 | 0.0039 | 45 |
| post_rerank_hit_at_5 | 0.9070 | 1.0000 | 0.0000 | 1.0000 | 43 |
| post_rerank_mrr_at_5 | 0.8012 | 1.0000 | 0.0000 | 1.0000 | 43 |
| post_rerank_ndcg_at_5 | 0.8132 | 1.0000 | 0.0000 | 1.0000 | 43 |
| post_rerank_recall_at_5 | 0.8798 | 1.0000 | 0.0000 | 1.0000 | 43 |
| precision_at_5 | 0.2605 | 0.6000 | 0.0000 | 0.8000 | 43 |
| r_precision | 0.7229 | 1.0000 | 0.0000 | 1.0000 | 43 |
| ranking_quality | 0.8063 | 1.0000 | 0.0000 | 1.0000 | 43 |
| recall_at_5 | 0.8178 | 1.0000 | 0.0000 | 1.0000 | 43 |
| refusal_correct | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 2 |
| reranker_latency | 4.5396 | 6.5119 | 2.6856 | 7.5380 | 45 |
| retrieval_latency | 0.1518 | 0.2553 | 0.1071 | 0.3438 | 45 |
| routing_accuracy | 0.9333 | 1.0000 | 0.0000 | 1.0000 | 45 |
| top1_cross_encoder_score | 0.6066 | 0.9991 | 0.0002 | 0.9995 | 43 |
| top3_cross_encoder_score | 0.3726 | 0.9557 | 0.0001 | 0.9759 | 43 |
| total_latency | 27.5120 | 31.2040 | 5.4782 | 40.3403 | 45 |
| unanswerable_retrieval_empty | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2 |
| visual_routing_accuracy | 0.9111 | 1.0000 | 0.0000 | 1.0000 | 45 |

## By Question Type

| Group | Metric | Mean |
|---|---|---|
| conceptual | answer_f1 | 0.2269 |
| conceptual | answer_length | 1694.2500 |
| conceptual | avg_cross_encoder_score | 0.1140 |
| conceptual | chunk_coverage | 0.4111 |
| conceptual | citation_count | 6.1667 |
| conceptual | citation_coverage | 0.9167 |
| conceptual | citation_provenance_precision | 1.0000 |
| conceptual | context_length | 5611.1667 |
| conceptual | context_recall | 0.9167 |
| conceptual | generation_latency | 24.0687 |
| conceptual | hit_at_5 | 0.9167 |
| conceptual | keyword_recall | 0.8819 |
| conceptual | lexical_content_jaccard | 0.1815 |
| conceptual | mrr | 0.9226 |
| conceptual | mrr_at_5 | 0.9167 |
| conceptual | ndcg_at_5 | 0.8971 |
| conceptual | planner_latency | 0.0004 |
| conceptual | post_rerank_hit_at_5 | 0.9167 |
| conceptual | post_rerank_mrr_at_5 | 0.8611 |
| conceptual | post_rerank_ndcg_at_5 | 0.8723 |
| conceptual | post_rerank_recall_at_5 | 0.9167 |
| conceptual | precision_at_5 | 0.2833 |
| conceptual | r_precision | 0.8889 |
| conceptual | ranking_quality | 0.8675 |
| conceptual | recall_at_5 | 0.8889 |
| conceptual | reranker_latency | 4.4453 |
| conceptual | retrieval_latency | 0.1405 |
| conceptual | routing_accuracy | 0.9167 |
| conceptual | top1_cross_encoder_score | 0.6109 |
| conceptual | top3_cross_encoder_score | 0.4067 |
| conceptual | total_latency | 28.6548 |
| conceptual | visual_routing_accuracy | 1.0000 |
| definition | answer_f1 | 0.4304 |
| definition | answer_length | 382.0000 |
| definition | avg_cross_encoder_score | 0.0067 |
| definition | chunk_coverage | 0.3333 |
| definition | citation_count | 5.0000 |
| definition | citation_coverage | 1.0000 |
| definition | citation_provenance_precision | 1.0000 |
| definition | context_length | 5954.0000 |
| definition | context_recall | 1.0000 |
| definition | generation_latency | 22.7999 |
| definition | hit_at_5 | 1.0000 |
| definition | keyword_recall | 1.0000 |
| definition | lexical_content_jaccard | 0.2500 |
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
| definition | reranker_latency | 6.7613 |
| definition | retrieval_latency | 0.1605 |
| definition | routing_accuracy | 1.0000 |
| definition | top1_cross_encoder_score | 0.0355 |
| definition | top3_cross_encoder_score | 0.0269 |
| definition | total_latency | 29.7219 |
| definition | visual_routing_accuracy | 1.0000 |
| factual | answer_f1 | 0.2152 |
| factual | answer_length | 975.9412 |
| factual | avg_cross_encoder_score | 0.1201 |
| factual | chunk_coverage | 0.4353 |
| factual | citation_count | 6.5294 |
| factual | citation_coverage | 0.9020 |
| factual | citation_provenance_precision | 1.0000 |
| factual | context_length | 5624.4706 |
| factual | context_recall | 0.9020 |
| factual | generation_latency | 21.9496 |
| factual | hit_at_5 | 0.9412 |
| factual | keyword_recall | 0.8578 |
| factual | lexical_content_jaccard | 0.1895 |
| factual | mrr | 0.7990 |
| factual | mrr_at_5 | 0.7990 |
| factual | ndcg_at_5 | 0.7729 |
| factual | planner_latency | 0.0005 |
| factual | post_rerank_hit_at_5 | 0.9412 |
| factual | post_rerank_mrr_at_5 | 0.8284 |
| factual | post_rerank_ndcg_at_5 | 0.8338 |
| factual | post_rerank_recall_at_5 | 0.9020 |
| factual | precision_at_5 | 0.2471 |
| factual | r_precision | 0.6765 |
| factual | ranking_quality | 0.8284 |
| factual | recall_at_5 | 0.8235 |
| factual | reranker_latency | 4.1738 |
| factual | retrieval_latency | 0.1364 |
| factual | routing_accuracy | 1.0000 |
| factual | top1_cross_encoder_score | 0.7183 |
| factual | top3_cross_encoder_score | 0.4241 |
| factual | total_latency | 26.2603 |
| factual | visual_routing_accuracy | 0.9412 |
| graph | answer_f1 | 0.4847 |
| graph | answer_length | 277.3333 |
| graph | avg_cross_encoder_score | 0.0270 |
| graph | chunk_coverage | 0.2444 |
| graph | citation_count | 3.6667 |
| graph | citation_coverage | 0.1667 |
| graph | citation_provenance_precision | 1.0000 |
| graph | context_length | 5472.6667 |
| graph | context_recall | 0.1667 |
| graph | generation_latency | 23.6778 |
| graph | hit_at_5 | 0.3333 |
| graph | keyword_recall | 1.0000 |
| graph | lexical_content_jaccard | 0.3084 |
| graph | mrr | 0.2560 |
| graph | mrr_at_5 | 0.1667 |
| graph | ndcg_at_5 | 0.1290 |
| graph | planner_latency | 0.0003 |
| graph | post_rerank_hit_at_5 | 0.6667 |
| graph | post_rerank_mrr_at_5 | 0.1778 |
| graph | post_rerank_ndcg_at_5 | 0.2311 |
| graph | post_rerank_recall_at_5 | 0.5000 |
| graph | precision_at_5 | 0.0667 |
| graph | r_precision | 0.1667 |
| graph | ranking_quality | 0.2016 |
| graph | recall_at_5 | 0.1667 |
| graph | reranker_latency | 4.4891 |
| graph | retrieval_latency | 0.2765 |
| graph | routing_accuracy | 1.0000 |
| graph | top1_cross_encoder_score | 0.1501 |
| graph | top3_cross_encoder_score | 0.1044 |
| graph | total_latency | 28.4437 |
| graph | visual_routing_accuracy | 1.0000 |
| unanswerable | answer_f1 | 0.0991 |
| unanswerable | answer_length | 39.0000 |
| unanswerable | chunk_coverage | 0.2667 |
| unanswerable | citation_count | 4.0000 |
| unanswerable | citation_provenance_precision | 1.0000 |
| unanswerable | context_length | 5484.5000 |
| unanswerable | generation_latency | 23.8599 |
| unanswerable | keyword_recall | 0.0000 |
| unanswerable | lexical_content_jaccard | 0.0426 |
| unanswerable | planner_latency | 0.0006 |
| unanswerable | refusal_correct | 1.0000 |
| unanswerable | reranker_latency | 4.4808 |
| unanswerable | retrieval_latency | 0.2189 |
| unanswerable | routing_accuracy | 1.0000 |
| unanswerable | total_latency | 28.5602 |
| unanswerable | unanswerable_retrieval_empty | 0.0000 |
| unanswerable | visual_routing_accuracy | 1.0000 |
| visual | answer_f1 | 0.3052 |
| visual | answer_length | 176.3000 |
| visual | avg_cross_encoder_score | 0.1048 |
| visual | chunk_coverage | 0.2600 |
| visual | citation_count | 3.9000 |
| visual | citation_coverage | 0.8417 |
| visual | citation_provenance_precision | 1.0000 |
| visual | context_length | 5335.3000 |
| visual | context_recall | 0.8417 |
| visual | generation_latency | 22.3383 |
| visual | hit_at_5 | 0.9000 |
| visual | keyword_recall | 0.6190 |
| visual | lexical_content_jaccard | 0.2316 |
| visual | mrr | 0.8611 |
| visual | mrr_at_5 | 0.8500 |
| visual | ndcg_at_5 | 0.8581 |
| visual | planner_latency | 0.0005 |
| visual | post_rerank_hit_at_5 | 0.9000 |
| visual | post_rerank_mrr_at_5 | 0.8500 |
| visual | post_rerank_ndcg_at_5 | 0.8631 |
| visual | post_rerank_recall_at_5 | 0.9000 |
| visual | precision_at_5 | 0.3200 |
| visual | r_precision | 0.7417 |
| visual | ranking_quality | 0.8571 |
| visual | recall_at_5 | 0.9000 |
| visual | reranker_latency | 5.0796 |
| visual | retrieval_latency | 0.1401 |
| visual | routing_accuracy | 0.8000 |
| visual | top1_cross_encoder_score | 0.6055 |
| visual | top3_cross_encoder_score | 0.3593 |
| visual | total_latency | 27.5584 |
| visual | visual_routing_accuracy | 0.7000 |

## By Difficulty

| Group | Metric | Mean |
|---|---|---|
| easy | answer_f1 | 0.2697 |
| easy | answer_length | 678.5357 |
| easy | avg_cross_encoder_score | 0.1071 |
| easy | chunk_coverage | 0.3810 |
| easy | citation_count | 5.7143 |
| easy | citation_coverage | 0.8839 |
| easy | citation_provenance_precision | 1.0000 |
| easy | context_length | 5529.9286 |
| easy | context_recall | 0.8839 |
| easy | generation_latency | 22.1811 |
| easy | hit_at_5 | 0.9286 |
| easy | keyword_recall | 0.8134 |
| easy | lexical_content_jaccard | 0.2215 |
| easy | mrr | 0.8284 |
| easy | mrr_at_5 | 0.8244 |
| easy | ndcg_at_5 | 0.8114 |
| easy | planner_latency | 0.0005 |
| easy | post_rerank_hit_at_5 | 0.9286 |
| easy | post_rerank_mrr_at_5 | 0.8601 |
| easy | post_rerank_ndcg_at_5 | 0.8634 |
| easy | post_rerank_recall_at_5 | 0.9048 |
| easy | precision_at_5 | 0.2714 |
| easy | r_precision | 0.7113 |
| easy | ranking_quality | 0.8627 |
| easy | recall_at_5 | 0.8571 |
| easy | reranker_latency | 4.4911 |
| easy | retrieval_latency | 0.1371 |
| easy | routing_accuracy | 0.9643 |
| easy | top1_cross_encoder_score | 0.6312 |
| easy | top3_cross_encoder_score | 0.3701 |
| easy | total_latency | 26.8099 |
| easy | visual_routing_accuracy | 0.8571 |
| medium | answer_f1 | 0.2332 |
| medium | answer_length | 1234.0000 |
| medium | avg_cross_encoder_score | 0.1030 |
| medium | chunk_coverage | 0.3451 |
| medium | citation_count | 5.1765 |
| medium | citation_coverage | 0.7667 |
| medium | citation_provenance_precision | 1.0000 |
| medium | context_length | 5576.8235 |
| medium | context_recall | 0.7667 |
| medium | generation_latency | 23.8725 |
| medium | hit_at_5 | 0.8000 |
| medium | keyword_recall | 0.7402 |
| medium | lexical_content_jaccard | 0.1632 |
| medium | mrr | 0.7893 |
| medium | mrr_at_5 | 0.7667 |
| medium | ndcg_at_5 | 0.7435 |
| medium | planner_latency | 0.0004 |
| medium | post_rerank_hit_at_5 | 0.8667 |
| medium | post_rerank_mrr_at_5 | 0.6911 |
| medium | post_rerank_ndcg_at_5 | 0.7195 |
| medium | post_rerank_recall_at_5 | 0.8333 |
| medium | precision_at_5 | 0.2400 |
| medium | r_precision | 0.7444 |
| medium | ranking_quality | 0.7010 |
| medium | recall_at_5 | 0.7444 |
| medium | refusal_correct | 1.0000 |
| medium | reranker_latency | 4.6195 |
| medium | retrieval_latency | 0.1760 |
| medium | routing_accuracy | 0.8824 |
| medium | top1_cross_encoder_score | 0.5605 |
| medium | top3_cross_encoder_score | 0.3774 |
| medium | total_latency | 28.6684 |
| medium | unanswerable_retrieval_empty | 0.0000 |
| medium | visual_routing_accuracy | 1.0000 |

## Per-Sample Results

| # | Type | Difficulty | Query | Route | MRR@5 | Recall@5 | Answer F1 | Citation | Total Lat (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | factual | easy | What is the instructor's name mentioned in the fir | vector_only | 0.500 | 1.000 | 0.103 | 1.000 | 7.20 |
| 2 | factual | easy | How many devices could the original ARPANET suppor | vector_only | 1.000 | 1.000 | 0.190 | 1.000 | 5.48 |
| 3 | conceptual | medium | What topics related to security will be covered in | vector_only | 1.000 | 1.000 | 0.237 | 1.000 | 19.59 |
| 4 | conceptual | medium | What are the main qualities highlighted in the lec | vector_only | 1.000 | 1.000 | 0.305 | 1.000 | 30.09 |
| 5 | factual | easy | Will the lecture videos be automatically captioned | vector_only | 1.000 | 1.000 | 0.571 | 1.000 | 28.28 |
| 6 | conceptual | medium | Why are multiple layers of abstraction necessary i | vector_only | 1.000 | 1.000 | 0.045 | 1.000 | 29.95 |
| 7 | visual | easy | According to the slide, which of the following is  | vector_only | 1.000 | 1.000 | 0.444 | 1.000 | 26.23 |
| 8 | visual | easy | Which hardware component is connected to the cache | vector_only | 1.000 | 1.000 | 0.059 | 0.667 | 27.09 |
| 9 | visual | easy | What role does the 'Referee' play in the operating | vector_only | 1.000 | 1.000 | 0.480 | 0.750 | 29.47 |
| 10 | visual | easy | According to the slide, what is the purpose of com | vector_only | 1.000 | 1.000 | 0.377 | 1.000 | 23.38 |
| 11 | graph | medium | What is the prerequisite relationship between Oper | graph_only | 0.000 | 0.000 | 0.419 | 0.500 | 28.09 |
| 12 | graph | medium | How is Process Abstraction related to Process in t | graph_only | 0.000 | 0.000 | 0.519 | 0.000 | 28.65 |
| 13 | graph | medium | Which subsystem does Kernel connect to via USED_BY | graph_only | 0.500 | 0.500 | 0.516 | 0.000 | 28.59 |
| 14 | unanswerable | medium | What direct prerequisite relationship exists betwe | graph_only | n/a | n/a | 0.093 | n/a | 28.13 |
| 15 | unanswerable | medium | How is Bell's Law a prerequisite for the implement | graph_only | n/a | n/a | 0.105 | n/a | 28.99 |
| 16 | factual | easy | Who is the instructor for MIT 6.S191? | vector_only | 1.000 | 1.000 | 0.200 | 1.000 | 27.44 |
| 17 | factual | easy | What are the infrastructure and cost requirements  | vector_only | 1.000 | 1.000 | 0.554 | 1.000 | 28.74 |
| 18 | conceptual | medium | According to the lecture, what is the difference b | vector_only | 1.000 | 1.000 | 0.136 | 1.000 | 29.25 |
| 19 | factual | easy | Which programming frameworks will be offered for t | vector_only | 1.000 | 1.000 | 0.429 | 1.000 | 26.02 |
| 20 | factual | easy | What are the three main factors that have enabled  | vector_only | 0.000 | 0.000 | 0.000 | 0.000 | 28.21 |
| 21 | conceptual | easy | What role does the bias term play in a neuron? | vector_only | 1.000 | 1.000 | 0.486 | 1.000 | 28.98 |
| 22 | conceptual | medium | Why are activation functions necessary in neural n | vector_only | 1.000 | 1.000 | 0.062 | 1.000 | 31.20 |
| 23 | conceptual | medium | What is the sign of the neuron's output for the po | vector_only | 1.000 | 1.000 | 0.448 | 1.000 | 27.03 |
| 24 | factual | easy | What are the three steps involved in the operation | vector_only | 1.000 | 1.000 | 0.279 | 1.000 | 40.34 |
| 25 | factual | easy | How do TensorFlow and PyTorch allow users to creat | vector_only | 1.000 | 1.000 | 0.227 | 1.000 | 17.79 |
| 26 | conceptual | medium | Why are fully connected layers stacked with nonlin | vector_only | 1.000 | 1.000 | 0.055 | 1.000 | 31.01 |
| 27 | factual | easy | What probability does the neural network predict f | vector_only | 1.000 | 0.500 | 0.500 | 1.000 | 26.97 |
| 28 | factual | easy | Why do classification networks output values betwe | vector_only | 0.500 | 1.000 | 0.087 | 1.000 | 29.95 |
| 29 | visual | easy | What device was used to run the live demonstration | vector_only | 1.000 | 1.000 | 0.000 | 1.000 | 28.33 |
| 30 | visual | easy | Which companies and institutions are listed as spo | vector_only | 1.000 | 1.000 | 0.571 | 1.000 | 28.17 |
| 31 | conceptual | medium | What are the three different contexts in which the | vector_only | 1.000 | 1.000 | 0.515 | 1.000 | 29.83 |
| 32 | conceptual | medium | Why is a static word embedding insufficient for re | vector_only | 1.000 | 1.000 | 0.113 | 1.000 | 29.48 |
| 33 | conceptual | medium | How does self-attention modify the embedding of a  | vector_only | 1.000 | 0.667 | 0.267 | 1.000 | 27.21 |
| 34 | definition | easy | According to the speaker, what determines the mean | vector_only | 1.000 | 1.000 | 0.430 | 1.000 | 29.72 |
| 35 | factual | easy | What effect does adding phone embeddings have on t | vector_only | 1.000 | 1.000 | 0.060 | 1.000 | 30.28 |
| 36 | conceptual | medium | Why do the coefficients used for combining word em | vector_only | 0.000 | 0.000 | 0.054 | 0.000 | 30.24 |
| 37 | factual | easy | How is the similarity between two words determined | vector_only | 1.000 | 1.000 | 0.048 | 1.000 | 30.36 |
| 38 | factual | easy | Which function is used to convert similarity score | vector_only | 1.000 | 1.000 | 0.118 | 1.000 | 27.32 |
| 39 | factual | easy | What is one limitation of using only fixed word em | vector_only | 1.000 | 0.667 | 0.232 | 1.000 | 29.60 |
| 40 | factual | easy | Why does the speaker want to introduce learnable p | vector_only | 0.250 | 0.500 | 0.053 | 1.000 | 31.13 |
| 41 | visual | easy | Which brand of microphone is visible in the podcas | vector_only | 0.000 | 0.000 | 0.133 | 0.000 | 26.42 |
| 42 | visual | easy | In the 2D embedding plot described, what do the X‑ | vector_only | 0.500 | 1.000 | 0.426 | 1.000 | 28.95 |
| 43 | factual | easy | Which function is used to normalize similarity sco | vector_only | 0.333 | 0.333 | 0.007 | 0.333 | 31.32 |
| 44 | visual | easy | In the educational slide, which Q vector is used w | vector_only | 1.000 | 1.000 | 0.486 | 1.000 | 27.51 |
| 45 | visual | medium | What feature is highlighted in the educational sli | vector_only | 1.000 | 1.000 | 0.074 | 1.000 | 30.03 |
