# LectureMind Evaluation Report
**Generated:** 20261010_113144
**Samples:** 45 (45 successful, 0 failed)

## Aggregated Metrics

| Metric | Mean | p95 | Min | Max | n |
|---|---|---|---|---|---|
| answer_f1 | 0.2667 | 0.5778 | 0.0000 | 0.6667 | 45 |
| answer_length | 806.6222 | 2663.0000 | 39.0000 | 2989.0000 | 45 |
| avg_cross_encoder_score | 0.1013 | 0.3386 | 0.0001 | 0.4623 | 45 |
| chunk_coverage | 0.2222 | 0.3333 | 0.0667 | 0.3333 | 45 |
| citation_count | 3.3333 | 5.0000 | 1.0000 | 5.0000 | 45 |
| citation_coverage | 0.7778 | 1.0000 | 0.0000 | 1.0000 | 45 |
| citation_provenance_precision | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 45 |
| context_length | 3281.3556 | 3891.0000 | 1521.0000 | 3973.0000 | 45 |
| generation_latency | 13.3101 | 16.4444 | 0.9471 | 29.3014 | 45 |
| hit_at_5 | 0.8667 | 1.0000 | 0.0000 | 1.0000 | 45 |
| keyword_recall | 0.7540 | 1.0000 | 0.0000 | 1.0000 | 45 |
| lexical_content_jaccard | 0.1935 | 0.4375 | 0.0000 | 0.5000 | 45 |
| mrr | 0.7107 | 1.0000 | 0.0000 | 1.0000 | 45 |
| mrr_at_5 | 0.7007 | 1.0000 | 0.0000 | 1.0000 | 45 |
| ndcg_at_5 | 0.7702 | 1.0000 | 0.0000 | 1.0000 | 45 |
| planner_latency | 0.0006 | 0.0010 | 0.0001 | 0.0034 | 45 |
| precision_at_5 | 0.1644 | 0.2000 | 0.0000 | 0.2000 | 45 |
| r_precision | 0.6333 | 1.0000 | 0.0000 | 1.0000 | 45 |
| ranking_quality | 0.7304 | 1.0000 | 0.0000 | 1.0000 | 45 |
| recall_at_5 | 0.8556 | 1.0000 | 0.0000 | 1.0000 | 45 |
| reranker_latency | 3.8772 | 5.1460 | 2.2292 | 5.4565 | 45 |
| retrieval_latency | 0.1322 | 0.2035 | 0.0978 | 0.2208 | 45 |
| routing_accuracy | 0.8889 | 1.0000 | 0.0000 | 1.0000 | 45 |
| top1_cross_encoder_score | 0.5818 | 0.9991 | 0.0002 | 0.9995 | 45 |
| top3_cross_encoder_score | 0.3572 | 0.9557 | 0.0001 | 0.9759 | 45 |
| total_latency | 17.3200 | 19.8641 | 4.3858 | 34.3662 | 45 |
| visual_routing_accuracy | 0.9111 | 1.0000 | 0.0000 | 1.0000 | 45 |

## By Question Type

| Group | Metric | Mean |
|---|---|---|
| conceptual | answer_f1 | 0.2385 |
| conceptual | answer_length | 1420.6667 |
| conceptual | avg_cross_encoder_score | 0.1140 |
| conceptual | chunk_coverage | 0.2556 |
| conceptual | citation_count | 3.8333 |
| conceptual | citation_coverage | 0.9167 |
| conceptual | citation_provenance_precision | 1.0000 |
| conceptual | context_length | 3274.7500 |
| conceptual | generation_latency | 12.2601 |
| conceptual | hit_at_5 | 0.9167 |
| conceptual | keyword_recall | 0.8889 |
| conceptual | lexical_content_jaccard | 0.2128 |
| conceptual | mrr | 0.9226 |
| conceptual | mrr_at_5 | 0.9167 |
| conceptual | ndcg_at_5 | 0.9167 |
| conceptual | planner_latency | 0.0005 |
| conceptual | precision_at_5 | 0.1833 |
| conceptual | r_precision | 0.9167 |
| conceptual | ranking_quality | 0.8259 |
| conceptual | recall_at_5 | 0.9167 |
| conceptual | reranker_latency | 3.7588 |
| conceptual | retrieval_latency | 0.1299 |
| conceptual | routing_accuracy | 0.8333 |
| conceptual | top1_cross_encoder_score | 0.6109 |
| conceptual | top3_cross_encoder_score | 0.4067 |
| conceptual | total_latency | 16.1492 |
| conceptual | visual_routing_accuracy | 1.0000 |
| definition | answer_f1 | 0.4474 |
| definition | answer_length | 344.0000 |
| definition | avg_cross_encoder_score | 0.0067 |
| definition | chunk_coverage | 0.2667 |
| definition | citation_count | 4.0000 |
| definition | citation_coverage | 1.0000 |
| definition | citation_provenance_precision | 1.0000 |
| definition | context_length | 3242.0000 |
| definition | generation_latency | 11.5836 |
| definition | hit_at_5 | 1.0000 |
| definition | keyword_recall | 1.0000 |
| definition | lexical_content_jaccard | 0.2414 |
| definition | mrr | 1.0000 |
| definition | mrr_at_5 | 1.0000 |
| definition | ndcg_at_5 | 1.0000 |
| definition | planner_latency | 0.0003 |
| definition | precision_at_5 | 0.2000 |
| definition | r_precision | 1.0000 |
| definition | ranking_quality | 1.0000 |
| definition | recall_at_5 | 1.0000 |
| definition | reranker_latency | 5.1400 |
| definition | retrieval_latency | 0.1083 |
| definition | routing_accuracy | 1.0000 |
| definition | top1_cross_encoder_score | 0.0355 |
| definition | top3_cross_encoder_score | 0.0269 |
| definition | total_latency | 16.8322 |
| definition | visual_routing_accuracy | 1.0000 |
| factual | answer_f1 | 0.2374 |
| factual | answer_length | 953.1176 |
| factual | avg_cross_encoder_score | 0.1201 |
| factual | chunk_coverage | 0.2667 |
| factual | citation_count | 4.0000 |
| factual | citation_coverage | 0.8235 |
| factual | citation_provenance_precision | 1.0000 |
| factual | context_length | 3434.1765 |
| factual | generation_latency | 13.2199 |
| factual | hit_at_5 | 0.8824 |
| factual | keyword_recall | 0.8039 |
| factual | lexical_content_jaccard | 0.1917 |
| factual | mrr | 0.7500 |
| factual | mrr_at_5 | 0.7500 |
| factual | ndcg_at_5 | 0.7837 |
| factual | planner_latency | 0.0007 |
| factual | precision_at_5 | 0.1765 |
| factual | r_precision | 0.6471 |
| factual | ranking_quality | 0.7961 |
| factual | recall_at_5 | 0.8824 |
| factual | reranker_latency | 3.5103 |
| factual | retrieval_latency | 0.1211 |
| factual | routing_accuracy | 0.9412 |
| factual | top1_cross_encoder_score | 0.7183 |
| factual | top3_cross_encoder_score | 0.4241 |
| factual | total_latency | 16.8520 |
| factual | visual_routing_accuracy | 0.9412 |
| graph | answer_f1 | 0.4838 |
| graph | answer_length | 245.0000 |
| graph | avg_cross_encoder_score | 0.0270 |
| graph | chunk_coverage | 0.0667 |
| graph | citation_count | 1.0000 |
| graph | citation_coverage | 0.0000 |
| graph | citation_provenance_precision | 1.0000 |
| graph | context_length | 2914.6667 |
| graph | generation_latency | 12.8125 |
| graph | hit_at_5 | 0.3333 |
| graph | keyword_recall | 0.8333 |
| graph | lexical_content_jaccard | 0.2980 |
| graph | mrr | 0.2560 |
| graph | mrr_at_5 | 0.1667 |
| graph | ndcg_at_5 | 0.1290 |
| graph | planner_latency | 0.0005 |
| graph | precision_at_5 | 0.0667 |
| graph | r_precision | 0.1667 |
| graph | ranking_quality | 0.2016 |
| graph | recall_at_5 | 0.1667 |
| graph | reranker_latency | 4.0462 |
| graph | retrieval_latency | 0.1982 |
| graph | routing_accuracy | 1.0000 |
| graph | top1_cross_encoder_score | 0.1501 |
| graph | top3_cross_encoder_score | 0.1044 |
| graph | total_latency | 17.0574 |
| graph | visual_routing_accuracy | 1.0000 |
| unanswerable | answer_f1 | 0.0991 |
| unanswerable | answer_length | 39.0000 |
| unanswerable | avg_cross_encoder_score | 0.0054 |
| unanswerable | chunk_coverage | 0.1667 |
| unanswerable | citation_count | 2.5000 |
| unanswerable | citation_coverage | 1.0000 |
| unanswerable | citation_provenance_precision | 1.0000 |
| unanswerable | context_length | 3510.5000 |
| unanswerable | generation_latency | 14.2216 |
| unanswerable | hit_at_5 | 1.0000 |
| unanswerable | keyword_recall | 0.0000 |
| unanswerable | lexical_content_jaccard | 0.0426 |
| unanswerable | mrr | 0.0000 |
| unanswerable | mrr_at_5 | 0.0000 |
| unanswerable | ndcg_at_5 | 1.0000 |
| unanswerable | planner_latency | 0.0005 |
| unanswerable | precision_at_5 | 0.0000 |
| unanswerable | r_precision | 0.0000 |
| unanswerable | ranking_quality | 0.0000 |
| unanswerable | recall_at_5 | 1.0000 |
| unanswerable | reranker_latency | 3.7449 |
| unanswerable | retrieval_latency | 0.1772 |
| unanswerable | routing_accuracy | 1.0000 |
| unanswerable | top1_cross_encoder_score | 0.0488 |
| unanswerable | top3_cross_encoder_score | 0.0263 |
| unanswerable | total_latency | 18.1442 |
| unanswerable | visual_routing_accuracy | 1.0000 |
| visual | answer_f1 | 0.3006 |
| visual | answer_length | 189.0000 |
| visual | avg_cross_encoder_score | 0.1048 |
| visual | chunk_coverage | 0.1600 |
| visual | citation_count | 2.4000 |
| visual | citation_coverage | 0.7000 |
| visual | citation_provenance_precision | 1.0000 |
| visual | context_length | 3097.6000 |
| visual | generation_latency | 14.8631 |
| visual | hit_at_5 | 0.9000 |
| visual | keyword_recall | 0.6098 |
| visual | lexical_content_jaccard | 0.1672 |
| visual | mrr | 0.6394 |
| visual | mrr_at_5 | 0.6283 |
| visual | ndcg_at_5 | 0.6948 |
| visual | planner_latency | 0.0006 |
| visual | precision_at_5 | 0.1800 |
| visual | r_precision | 0.5000 |
| visual | ranking_quality | 0.7821 |
| visual | recall_at_5 | 0.9000 |
| visual | reranker_latency | 4.4923 |
| visual | retrieval_latency | 0.1273 |
| visual | routing_accuracy | 0.8000 |
| visual | top1_cross_encoder_score | 0.6055 |
| visual | top3_cross_encoder_score | 0.3593 |
| visual | total_latency | 19.4833 |
| visual | visual_routing_accuracy | 0.7000 |

## By Difficulty

| Group | Metric | Mean |
|---|---|---|
| easy | answer_f1 | 0.2744 |
| easy | answer_length | 661.9643 |
| easy | avg_cross_encoder_score | 0.1071 |
| easy | chunk_coverage | 0.2381 |
| easy | citation_count | 3.5714 |
| easy | citation_coverage | 0.8214 |
| easy | citation_provenance_precision | 1.0000 |
| easy | context_length | 3393.1786 |
| easy | generation_latency | 14.0140 |
| easy | hit_at_5 | 0.8929 |
| easy | keyword_recall | 0.7684 |
| easy | lexical_content_jaccard | 0.1994 |
| easy | mrr | 0.7194 |
| easy | mrr_at_5 | 0.7155 |
| easy | ndcg_at_5 | 0.7597 |
| easy | planner_latency | 0.0006 |
| easy | precision_at_5 | 0.1786 |
| easy | r_precision | 0.6071 |
| easy | ranking_quality | 0.8162 |
| easy | recall_at_5 | 0.8929 |
| easy | reranker_latency | 3.8356 |
| easy | retrieval_latency | 0.1227 |
| easy | routing_accuracy | 0.9286 |
| easy | top1_cross_encoder_score | 0.6312 |
| easy | top3_cross_encoder_score | 0.3701 |
| easy | total_latency | 17.9729 |
| easy | visual_routing_accuracy | 0.8571 |
| medium | answer_f1 | 0.2539 |
| medium | answer_length | 1044.8824 |
| medium | avg_cross_encoder_score | 0.0915 |
| medium | chunk_coverage | 0.1961 |
| medium | citation_count | 2.9412 |
| medium | citation_coverage | 0.7059 |
| medium | citation_provenance_precision | 1.0000 |
| medium | context_length | 3097.1765 |
| medium | generation_latency | 12.1507 |
| medium | hit_at_5 | 0.8235 |
| medium | keyword_recall | 0.7304 |
| medium | lexical_content_jaccard | 0.1837 |
| medium | mrr | 0.6964 |
| medium | mrr_at_5 | 0.6765 |
| medium | ndcg_at_5 | 0.7875 |
| medium | planner_latency | 0.0005 |
| medium | precision_at_5 | 0.1412 |
| medium | r_precision | 0.6765 |
| medium | ranking_quality | 0.5891 |
| medium | recall_at_5 | 0.7941 |
| medium | reranker_latency | 3.9457 |
| medium | retrieval_latency | 0.1477 |
| medium | routing_accuracy | 0.8235 |
| medium | top1_cross_encoder_score | 0.5003 |
| medium | top3_cross_encoder_score | 0.3361 |
| medium | total_latency | 16.2446 |
| medium | visual_routing_accuracy | 1.0000 |

## Per-Sample Results

| # | Type | Difficulty | Query | Route | MRR@5 | Recall@5 | Answer F1 | Citation | Total Lat (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | factual | easy | What is the instructor's name mentioned in the fir | vector_only | 0.500 | 1.000 | 0.308 | 1.000 | 5.48 |
| 2 | factual | easy | How many devices could the original ARPANET suppor | vector_only | 1.000 | 1.000 | 0.235 | 1.000 | 4.39 |
| 3 | conceptual | medium | What topics related to security will be covered in | vector_only | 1.000 | 1.000 | 0.306 | 1.000 | 5.44 |
| 4 | conceptual | medium | What are the main qualities highlighted in the lec | vector_only | 1.000 | 1.000 | 0.396 | 1.000 | 5.33 |
| 5 | factual | easy | Will the lecture videos be automatically captioned | vector_only | 1.000 | 1.000 | 0.542 | 1.000 | 14.07 |
| 6 | conceptual | medium | Why are multiple layers of abstraction necessary i | vector_only | 1.000 | 1.000 | 0.054 | 1.000 | 19.23 |
| 7 | visual | easy | According to the slide, which of the following is  | vector_only | 1.000 | 1.000 | 0.346 | 1.000 | 18.25 |
| 8 | visual | easy | Which hardware component is connected to the cache | vector_only | 0.250 | 1.000 | 0.054 | 1.000 | 19.13 |
| 9 | visual | easy | What role does the 'Referee' play in the operating | vector_only | 0.200 | 1.000 | 0.458 | 0.000 | 18.57 |
| 10 | visual | easy | According to the slide, what is the purpose of com | vector_only | 0.333 | 1.000 | 0.444 | 1.000 | 19.60 |
| 11 | graph | medium | What is the prerequisite relationship between Oper | graph_only | 0.000 | 0.000 | 0.417 | 0.000 | 17.46 |
| 12 | graph | medium | How is Process Abstraction related to Process in t | graph_only | 0.000 | 0.000 | 0.618 | 0.000 | 16.55 |
| 13 | graph | medium | Which subsystem does Kernel connect to via USED_BY | graph_only | 0.500 | 0.500 | 0.417 | 0.000 | 17.16 |
| 14 | unanswerable | medium | What direct prerequisite relationship exists betwe | graph_only | 0.000 | 1.000 | 0.093 | 1.000 | 19.14 |
| 15 | unanswerable | medium | How is Bell's Law a prerequisite for the implement | graph_only | 0.000 | 1.000 | 0.105 | 1.000 | 17.15 |
| 16 | factual | easy | Who is the instructor for MIT 6.S191? | vector_only | 1.000 | 1.000 | 0.114 | 1.000 | 17.96 |
| 17 | factual | easy | What are the infrastructure and cost requirements  | vector_only | 1.000 | 1.000 | 0.571 | 1.000 | 19.34 |
| 18 | conceptual | medium | According to the lecture, what is the difference b | vector_only | 1.000 | 1.000 | 0.135 | 1.000 | 18.52 |
| 19 | factual | easy | Which programming frameworks will be offered for t | vector_only | 1.000 | 1.000 | 0.375 | 1.000 | 16.81 |
| 20 | factual | easy | What are the three main factors that have enabled  | vector_only | 0.000 | 0.000 | 0.000 | 0.000 | 16.60 |
| 21 | conceptual | easy | What role does the bias term play in a neuron? | vector_only | 1.000 | 1.000 | 0.391 | 1.000 | 19.25 |
| 22 | conceptual | medium | Why are activation functions necessary in neural n | vector_only | 1.000 | 1.000 | 0.071 | 1.000 | 18.63 |
| 23 | conceptual | medium | What is the sign of the neuron's output for the po | vector_only | 1.000 | 1.000 | 0.459 | 1.000 | 17.57 |
| 24 | factual | easy | What are the three steps involved in the operation | vector_only | 1.000 | 1.000 | 0.415 | 1.000 | 17.24 |
| 25 | factual | easy | How do TensorFlow and PyTorch allow users to creat | vector_only | 1.000 | 1.000 | 0.241 | 1.000 | 17.52 |
| 26 | conceptual | medium | Why are fully connected layers stacked with nonlin | vector_only | 1.000 | 1.000 | 0.064 | 1.000 | 19.15 |
| 27 | factual | easy | What probability does the neural network predict f | vector_only | 1.000 | 1.000 | 0.578 | 1.000 | 18.57 |
| 28 | factual | easy | Why do classification networks output values betwe | vector_only | 0.500 | 1.000 | 0.085 | 1.000 | 18.58 |
| 29 | visual | easy | What device was used to run the live demonstration | vector_only | 1.000 | 1.000 | 0.000 | 1.000 | 19.36 |
| 30 | visual | easy | Which companies and institutions are listed as spo | vector_only | 1.000 | 1.000 | 0.667 | 1.000 | 16.50 |
| 31 | conceptual | medium | What are the three different contexts in which the | vector_only | 1.000 | 1.000 | 0.462 | 1.000 | 15.08 |
| 32 | conceptual | medium | Why is a static word embedding insufficient for re | vector_only | 1.000 | 1.000 | 0.118 | 1.000 | 19.68 |
| 33 | conceptual | medium | How does self-attention modify the embedding of a  | vector_only | 1.000 | 1.000 | 0.331 | 1.000 | 17.76 |
| 34 | definition | easy | According to the speaker, what determines the mean | vector_only | 1.000 | 1.000 | 0.447 | 1.000 | 16.83 |
| 35 | factual | easy | What effect does adding phone embeddings have on t | vector_only | 1.000 | 1.000 | 0.083 | 1.000 | 19.15 |
| 36 | conceptual | medium | Why do the coefficients used for combining word em | vector_only | 0.000 | 0.000 | 0.074 | 0.000 | 18.15 |
| 37 | factual | easy | How is the similarity between two words determined | vector_only | 1.000 | 1.000 | 0.058 | 1.000 | 19.26 |
| 38 | factual | easy | Which function is used to convert similarity score | vector_only | 0.500 | 1.000 | 0.084 | 1.000 | 18.03 |
| 39 | factual | easy | What is one limitation of using only fixed word em | vector_only | 1.000 | 1.000 | 0.277 | 1.000 | 18.65 |
| 40 | factual | easy | Why does the speaker want to introduce learnable p | vector_only | 0.250 | 1.000 | 0.063 | 0.000 | 24.98 |
| 41 | visual | easy | Which brand of microphone is visible in the podcas | vector_only | 0.000 | 0.000 | 0.133 | 0.000 | 34.37 |
| 42 | visual | easy | In the 2D embedding plot described, what do the X‑ | vector_only | 0.500 | 1.000 | 0.366 | 1.000 | 18.13 |
| 43 | factual | easy | Which function is used to normalize similarity sco | vector_only | 0.000 | 0.000 | 0.007 | 0.000 | 19.86 |
| 44 | visual | easy | In the educational slide, which Q vector is used w | vector_only | 1.000 | 1.000 | 0.340 | 1.000 | 16.78 |
| 45 | visual | medium | What feature is highlighted in the educational sli | vector_only | 1.000 | 1.000 | 0.197 | 0.000 | 14.16 |
