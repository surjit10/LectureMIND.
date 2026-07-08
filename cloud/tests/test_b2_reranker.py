# cloud/tests/test_b2_reranker.py
#
# REMOVED — Stage B2 (Per-Lecture Reranker Training) has been eliminated.
#
# Architecture change (see migration_analysis_report.md):
#   - Knowledge packages no longer contain a reranker_model/ directory.
#   - Per-lecture fine-tuning (cloud/training/reranker_trainer.py) is no
#     longer invoked by the cloud pipeline orchestrator.
#   - Stage B1 (triplet generation) still runs and writes triplets.json,
#     which is used for offline global reranker training.
#   - The global reranker is trained once, stored at:
#       local_runtime/models/global_reranker/
#     and loaded as an application singleton by serving/fastapi/app.py.
#
# If you need to test global reranker training, create a dedicated offline
# training script and test suite outside of the cloud pipeline.

