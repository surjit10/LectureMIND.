# LectureMind Evaluation Report
**Generated:** 20260811_105621
**Samples:** 50 (50 successful, 0 failed)

## Aggregated Metrics

| Metric | Mean | p95 | Min | Max | n |
|---|---|---|---|---|---|
| answer_f1 | 0.4594 | 0.7143 | 0.0000 | 0.8800 | 50 |
| answer_length | 485.7600 | 1606.0000 | 39.0000 | 2475.0000 | 50 |
| answer_similarity | 0.3285 | 0.5854 | 0.0000 | 0.8485 | 50 |
| avg_cross_encoder_score | 0.0860 | 0.2125 | 0.0005 | 0.4923 | 50 |
| chunk_coverage | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 50 |
| citation_completeness | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 50 |
| citation_count | 15.0000 | 15.0000 | 15.0000 | 15.0000 | 50 |
| citation_coverage | 0.9267 | 1.0000 | 0.3333 | 1.0000 | 50 |
| context_length | 3459.5800 | 3915.0000 | 2509.0000 | 3954.0000 | 50 |
| generation_latency | 16.5648 | 18.9696 | 0.8327 | 19.5663 | 50 |
| hit_at_5 | 0.9800 | 1.0000 | 0.0000 | 1.0000 | 50 |
| keyword_recall | 0.5453 | 1.0000 | 0.0000 | 1.0000 | 50 |
| mrr | 0.7882 | 1.0000 | 0.1429 | 1.0000 | 50 |
| ndcg_at_5 | 0.7674 | 1.0000 | 0.0000 | 1.0000 | 50 |
| planner_latency | 0.0004 | 0.0008 | 0.0001 | 0.0036 | 50 |
| precision_at_5 | 0.2800 | 0.4000 | 0.0000 | 0.8000 | 50 |
| ranking_quality | 0.9183 | 1.0000 | 0.1667 | 1.0000 | 50 |
| recall_at_5 | 0.8620 | 1.0000 | 0.0000 | 1.0000 | 50 |
| reranker_latency | 3.8180 | 5.3341 | 2.9365 | 5.5313 | 50 |
| retrieval_latency | 0.1304 | 0.1716 | 0.0983 | 0.2655 | 50 |
| routing_accuracy | 0.9800 | 1.0000 | 0.0000 | 1.0000 | 50 |
| total_latency | 20.5137 | 22.8164 | 4.3215 | 23.1517 | 50 |
| visual_routing_accuracy | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 50 |

## By Question Type

| Group | Metric | Mean |
|---|---|---|
| conceptual | answer_f1 | 0.4055 |
| conceptual | answer_length | 679.2941 |
| conceptual | answer_similarity | 0.2783 |
| conceptual | avg_cross_encoder_score | 0.1009 |
| conceptual | chunk_coverage | 1.0000 |
| conceptual | citation_completeness | 1.0000 |
| conceptual | citation_count | 15.0000 |
| conceptual | citation_coverage | 0.8824 |
| conceptual | context_length | 3510.0000 |
| conceptual | generation_latency | 16.9068 |
| conceptual | hit_at_5 | 0.9412 |
| conceptual | keyword_recall | 0.4373 |
| conceptual | mrr | 0.8045 |
| conceptual | ndcg_at_5 | 0.7096 |
| conceptual | planner_latency | 0.0005 |
| conceptual | precision_at_5 | 0.3176 |
| conceptual | ranking_quality | 0.9706 |
| conceptual | recall_at_5 | 0.7412 |
| conceptual | reranker_latency | 3.7600 |
| conceptual | retrieval_latency | 0.1400 |
| conceptual | routing_accuracy | 0.9412 |
| conceptual | total_latency | 20.8074 |
| conceptual | visual_routing_accuracy | 1.0000 |
| definition | answer_f1 | 0.6315 |
| definition | answer_length | 340.0000 |
| definition | answer_similarity | 0.5059 |
| definition | avg_cross_encoder_score | 0.0637 |
| definition | chunk_coverage | 1.0000 |
| definition | citation_completeness | 1.0000 |
| definition | citation_count | 15.0000 |
| definition | citation_coverage | 1.0000 |
| definition | context_length | 3212.0000 |
| definition | generation_latency | 13.2589 |
| definition | hit_at_5 | 1.0000 |
| definition | keyword_recall | 0.6875 |
| definition | mrr | 0.7500 |
| definition | ndcg_at_5 | 0.8155 |
| definition | planner_latency | 0.0011 |
| definition | precision_at_5 | 0.2500 |
| definition | ranking_quality | 1.0000 |
| definition | recall_at_5 | 1.0000 |
| definition | reranker_latency | 3.6220 |
| definition | retrieval_latency | 0.1331 |
| definition | routing_accuracy | 1.0000 |
| definition | total_latency | 17.0152 |
| definition | visual_routing_accuracy | 1.0000 |
| factual | answer_f1 | 0.4223 |
| factual | answer_length | 385.6087 |
| factual | answer_similarity | 0.2821 |
| factual | avg_cross_encoder_score | 0.0753 |
| factual | chunk_coverage | 1.0000 |
| factual | citation_completeness | 1.0000 |
| factual | citation_count | 15.0000 |
| factual | citation_coverage | 0.9275 |
| factual | context_length | 3494.6087 |
| factual | generation_latency | 16.8986 |
| factual | hit_at_5 | 1.0000 |
| factual | keyword_recall | 0.6014 |
| factual | mrr | 0.8435 |
| factual | ndcg_at_5 | 0.8257 |
| factual | planner_latency | 0.0003 |
| factual | precision_at_5 | 0.2696 |
| factual | ranking_quality | 0.8442 |
| factual | recall_at_5 | 0.8913 |
| factual | reranker_latency | 3.6132 |
| factual | retrieval_latency | 0.1242 |
| factual | routing_accuracy | 1.0000 |
| factual | total_latency | 20.6364 |
| factual | visual_routing_accuracy | 1.0000 |
| summary | answer_f1 | 0.6822 |
| summary | answer_length | 577.0000 |
| summary | answer_similarity | 0.5634 |
| summary | avg_cross_encoder_score | 0.1789 |
| summary | chunk_coverage | 1.0000 |
| summary | citation_completeness | 1.0000 |
| summary | citation_count | 15.0000 |
| summary | citation_coverage | 1.0000 |
| summary | context_length | 3394.0000 |
| summary | generation_latency | 18.4415 |
| summary | hit_at_5 | 1.0000 |
| summary | keyword_recall | 0.2500 |
| summary | mrr | 0.5000 |
| summary | ndcg_at_5 | 0.6934 |
| summary | planner_latency | 0.0001 |
| summary | precision_at_5 | 0.4000 |
| summary | ranking_quality | 1.0000 |
| summary | recall_at_5 | 1.0000 |
| summary | reranker_latency | 3.2144 |
| summary | retrieval_latency | 0.1267 |
| summary | routing_accuracy | 1.0000 |
| summary | total_latency | 21.7827 |
| summary | visual_routing_accuracy | 1.0000 |
| visual | answer_f1 | 0.6309 |
| visual | answer_length | 386.8000 |
| visual | answer_similarity | 0.5236 |
| visual | avg_cross_encoder_score | 0.0834 |
| visual | chunk_coverage | 1.0000 |
| visual | citation_completeness | 1.0000 |
| visual | citation_count | 15.0000 |
| visual | citation_coverage | 1.0000 |
| visual | context_length | 3338.2000 |
| visual | generation_latency | 16.1362 |
| visual | hit_at_5 | 1.0000 |
| visual | keyword_recall | 0.6000 |
| visual | mrr | 0.5667 |
| visual | ndcg_at_5 | 0.6723 |
| visual | planner_latency | 0.0003 |
| visual | precision_at_5 | 0.2000 |
| visual | ranking_quality | 1.0000 |
| visual | recall_at_5 | 1.0000 |
| visual | reranker_latency | 5.2343 |
| visual | retrieval_latency | 0.1253 |
| visual | routing_accuracy | 1.0000 |
| visual | total_latency | 21.4962 |
| visual | visual_routing_accuracy | 1.0000 |

## By Difficulty

| Group | Metric | Mean |
|---|---|---|
| easy | answer_f1 | 0.4323 |
| easy | answer_length | 372.7727 |
| easy | answer_similarity | 0.3033 |
| easy | avg_cross_encoder_score | 0.0753 |
| easy | chunk_coverage | 1.0000 |
| easy | citation_completeness | 1.0000 |
| easy | citation_count | 15.0000 |
| easy | citation_coverage | 0.9167 |
| easy | context_length | 3417.7273 |
| easy | generation_latency | 16.4740 |
| easy | hit_at_5 | 1.0000 |
| easy | keyword_recall | 0.6174 |
| easy | mrr | 0.8250 |
| easy | ndcg_at_5 | 0.7746 |
| easy | planner_latency | 0.0003 |
| easy | precision_at_5 | 0.2909 |
| easy | ranking_quality | 0.8371 |
| easy | recall_at_5 | 0.8394 |
| easy | reranker_latency | 3.6682 |
| easy | retrieval_latency | 0.1334 |
| easy | routing_accuracy | 0.9545 |
| easy | total_latency | 20.2761 |
| easy | visual_routing_accuracy | 1.0000 |
| hard | answer_f1 | 0.5588 |
| hard | answer_length | 423.5000 |
| hard | answer_similarity | 0.3862 |
| hard | avg_cross_encoder_score | 0.0896 |
| hard | chunk_coverage | 1.0000 |
| hard | citation_completeness | 1.0000 |
| hard | citation_count | 15.0000 |
| hard | citation_coverage | 0.8333 |
| hard | context_length | 3598.5000 |
| hard | generation_latency | 18.6828 |
| hard | hit_at_5 | 1.0000 |
| hard | keyword_recall | 0.3333 |
| hard | mrr | 1.0000 |
| hard | ndcg_at_5 | 0.8827 |
| hard | planner_latency | 0.0007 |
| hard | precision_at_5 | 0.4000 |
| hard | ranking_quality | 1.0000 |
| hard | recall_at_5 | 0.8333 |
| hard | reranker_latency | 3.4284 |
| hard | retrieval_latency | 0.1495 |
| hard | routing_accuracy | 1.0000 |
| hard | total_latency | 22.2617 |
| hard | visual_routing_accuracy | 1.0000 |
| medium | answer_f1 | 0.4746 |
| medium | answer_length | 586.1538 |
| medium | answer_similarity | 0.3454 |
| medium | avg_cross_encoder_score | 0.0948 |
| medium | chunk_coverage | 1.0000 |
| medium | citation_completeness | 1.0000 |
| medium | citation_count | 15.0000 |
| medium | citation_coverage | 0.9423 |
| medium | context_length | 3484.3077 |
| medium | generation_latency | 16.4787 |
| medium | hit_at_5 | 0.9615 |
| medium | keyword_recall | 0.5006 |
| medium | mrr | 0.7408 |
| medium | ndcg_at_5 | 0.7524 |
| medium | planner_latency | 0.0005 |
| medium | precision_at_5 | 0.2615 |
| medium | ranking_quality | 0.9808 |
| medium | recall_at_5 | 0.8833 |
| medium | reranker_latency | 3.9747 |
| medium | retrieval_latency | 0.1264 |
| medium | routing_accuracy | 1.0000 |
| medium | total_latency | 20.5804 |
| medium | visual_routing_accuracy | 1.0000 |

## Per-Sample Results

| # | Type | Difficulty | Query | Route | MRR | Recall@5 | Answer F1 | Citation | Total Lat (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | definition | medium | What is the Global Data Plane project? | vector_only | 1.000 | 1.000 | 0.716 | 1.000 | 5.70 |
| 2 | factual | easy | How many devices could the original ARPANET handle | vector_only | 1.000 | 1.000 | 0.367 | 1.000 | 4.32 |
| 3 | conceptual | medium | What is the main challenge of operating systems th | vector_only | 0.500 | 1.000 | 0.385 | 1.000 | 5.71 |
| 4 | factual | easy | What does the lecture say about how many processes | vector_only | 1.000 | 1.000 | 0.466 | 1.000 | 14.64 |
| 5 | factual | easy | What did the lecture say about SSL and security mo | vector_only | 0.500 | 1.000 | 0.347 | 1.000 | 22.64 |
| 6 | factual | easy | How many processors does a modern car have accordi | vector_only | 0.500 | 0.500 | 0.333 | 1.000 | 22.71 |
| 7 | factual | easy | Which topics will the class cover according to the | vector_only | 0.200 | 0.333 | 0.263 | 0.667 | 20.88 |
| 8 | factual | easy | What did the lecture say about the honor code? | vector_only | 1.000 | 1.000 | 0.469 | 1.000 | 21.66 |
| 9 | factual | easy | What is the enrollment limit for CS162? | vector_only | 1.000 | 1.000 | 0.478 | 1.000 | 21.67 |
| 10 | conceptual | medium | What is the challenge of testing software for all  | vector_only | 1.000 | 1.000 | 0.348 | 1.000 | 22.86 |
| 11 | summary | medium | What is the syllabus of CS162 according to the lec | vector_only | 0.500 | 1.000 | 0.682 | 1.000 | 21.78 |
| 12 | definition | medium | What does the lecture say about the word 'system'? | vector_only | 0.500 | 1.000 | 0.763 | 1.000 | 21.66 |
| 13 | factual | easy | What did the lecture say about the Intel Skylake e | vector_only | 1.000 | 1.000 | 0.406 | 1.000 | 22.76 |
| 14 | definition | medium | What is Alewife? | vector_only | 0.500 | 1.000 | 0.559 | 1.000 | 19.45 |
| 15 | factual | easy | What does the lecture say about the number of comp | vector_only | 1.000 | 1.000 | 0.442 | 1.000 | 21.68 |
| 16 | factual | medium | How does a DNS lookup work when you use your phone | vector_only | 1.000 | 1.000 | 0.530 | 1.000 | 20.73 |
| 17 | conceptual | medium | Why must the OS provide a consistent programming a | vector_only | 0.143 | 0.000 | 0.155 | 1.000 | 20.40 |
| 18 | factual | easy | What are the requirements for group projects in CS | vector_only | 1.000 | 1.000 | 0.481 | 1.000 | 20.14 |
| 19 | factual | easy | What happens when a program becomes a process? | vector_only | 1.000 | 1.000 | 0.231 | 1.000 | 21.26 |
| 20 | factual | medium | What does the lecture say about the Mars Rover and | vector_only | 1.000 | 1.000 | 0.527 | 1.000 | 22.32 |
| 21 | conceptual | medium | What does the lecture say about files being an abs | vector_only | 0.333 | 0.333 | 0.418 | 0.333 | 22.82 |
| 22 | conceptual | easy | How does the lecture describe networking as part o | vector_only | 1.000 | 1.000 | 0.455 | 1.000 | 20.76 |
| 23 | conceptual | medium | Why does the lecture describe the internet as look | vector_only | 1.000 | 1.000 | 0.212 | 1.000 | 22.90 |
| 24 | factual | easy | What homework should students start right away? | vector_only | 0.500 | 1.000 | 0.595 | 1.000 | 21.42 |
| 25 | factual | medium | What is the power density problem described in the | vector_only | 1.000 | 1.000 | 0.504 | 1.000 | 21.73 |
| 26 | conceptual | medium | What collaboration is allowed and not allowed betw | vector_only | 1.000 | 1.000 | 0.404 | 1.000 | 22.44 |
| 27 | factual | medium | What does the lecture say about time scales like n | vector_only | 1.000 | 1.000 | 0.281 | 1.000 | 21.19 |
| 28 | conceptual | medium | What did the lecture say about the cloud and cloud | vector_only | 1.000 | 1.000 | 0.452 | 1.000 | 22.50 |
| 29 | factual | easy | What does the lecture say about caches? | vector_only | 1.000 | 1.000 | 0.484 | 1.000 | 22.52 |
| 30 | factual | easy | What is the course website and where should studen | vector_only | 1.000 | 0.667 | 0.486 | 0.667 | 20.49 |
| 31 | conceptual | medium | What role does virtual memory play in protecting p | vector_only | 1.000 | 0.667 | 0.422 | 1.000 | 22.16 |
| 32 | conceptual | medium | What does the lecture say about the OS providing c | vector_only | 0.500 | 1.000 | 0.424 | 1.000 | 21.58 |
| 33 | conceptual | medium | What is the kernel and why do people disagree abou | vector_only | 1.000 | 0.667 | 0.171 | 0.667 | 22.04 |
| 34 | conceptual | easy | What does the lecture say about how to think like  | vector_only | 0.200 | 0.333 | 0.310 | 0.333 | 19.71 |
| 35 | factual | medium | Who is the instructor of CS162 and what is his bac | vector_only | 0.500 | 1.000 | 0.667 | 1.000 | 22.28 |
| 36 | factual | medium | What does the lecture say about Linux and other sy | vector_only | 0.200 | 0.500 | 0.452 | 0.500 | 21.51 |
| 37 | conceptual | hard | What does the lecture say about whether the window | vector_only | 1.000 | 1.000 | 0.550 | 1.000 | 22.25 |
| 38 | conceptual | easy | What did the lecture say about cameras being requi | vector_only | 1.000 | 0.333 | 0.438 | 1.000 | 21.50 |
| 39 | definition | easy | What is the definition of an operating system acco | vector_only | 1.000 | 1.000 | 0.488 | 1.000 | 21.25 |
| 40 | factual | easy | What is the grading breakdown for CS162? | vector_only | 1.000 | 0.500 | 0.000 | 0.500 | 21.04 |
| 41 | factual | medium | Where does the word 'operating' come from accordin | vector_only | 1.000 | 1.000 | 0.460 | 1.000 | 22.44 |
| 42 | conceptual | hard | How does the operating system switch between two p | vector_only | 1.000 | 0.667 | 0.567 | 0.667 | 22.27 |
| 43 | factual | easy | What did the lecture say about the course website  | vector_only | 1.000 | 1.000 | 0.444 | 1.000 | 22.62 |
| 44 | conceptual | medium | What abstractions and higher-level objects does th | vector_only | 1.000 | 0.800 | 0.536 | 1.000 | 21.30 |
| 45 | conceptual | easy | What does the lecture say about the OS being an il | vector_only | 1.000 | 0.800 | 0.647 | 1.000 | 20.52 |
| 46 | visual | medium | What does the slide about network capacity show? | vector_only | 1.000 | 1.000 | 0.475 | 1.000 | 21.52 |
| 47 | visual | medium | What does the slide say about societal-scale infor | vector_only | 1.000 | 1.000 | 0.714 | 1.000 | 20.69 |
| 48 | visual | easy | According to the slide, what does a process consis | vector_only | 0.250 | 1.000 | 0.880 | 1.000 | 19.88 |
| 49 | visual | medium | What does the diagram of a compiled program's view | vector_only | 0.250 | 1.000 | 0.548 | 1.000 | 22.24 |
| 50 | visual | medium | What does the protection slide show about processe | vector_only | 0.333 | 1.000 | 0.537 | 1.000 | 23.15 |
