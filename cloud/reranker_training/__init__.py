# cloud/reranker_training/__init__.py
# Pipeline B — Global Reranker Training (independent Kaggle workflow).
#
# COMPLETELY separate from the lecture processing pipeline (Pipeline A):
#   - Pipeline A (cloud/orchestration/run_ingestion_pipeline.py) produces
#     knowledge packages. It NEVER trains models.
#   - Pipeline B consumes ONLY the triplets.json files inside previously
#     exported knowledge packages. It NEVER processes videos.
# The only shared artifact between the two Kaggle workflows is triplets.json.
