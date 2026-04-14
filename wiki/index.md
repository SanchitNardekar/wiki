```markdown
# Wiki Index

A practical, production-oriented knowledge base covering ranking/retrieval, recommender systems, LLMs/RAG, and the engineering foundations to build and operate ML systems at scale.

## Summary Table

| Page | Key | Summary | Tags |
|---|---|---|---|
| [Learning to Rank](learning_to_rank.md) | `learning_to_rank` | Learning to Rank — LambdaMART, RankNet, pointwise/pairwise/listwise approaches | ranking, learning-to-rank, lambdamart, ranknet |
| [Recommendation Systems](recommendation_systems.md) | `recommendation_systems` | Recommendation Systems — Collaborative filtering, content-based, hybrid approaches | recommender-systems, collaborative-filtering, hybrid |
| [Two-Tower Architecture](two_tower_architecture.md) | `two_tower_architecture` | Two-Tower Architecture — Industry standard dual-encoder for retrieval and pre-ranking | retrieval, dual-encoder, embeddings |
| [Multi-Stage Ranking Pipelines](multi_stage_ranking.md) | `multi_stage_ranking` | Multi-Stage Ranking Pipelines — Retrieval to L1 to L2 to re-ranking funnel and latency budgets | ranking, pipelines, latency, production |
| [Search and Information Retrieval](search_and_retrieval.md) | `search_and_retrieval` | Search and Information Retrieval — BM25, TF-IDF, inverted indexes, query understanding | search, IR, bm25, tf-idf |
| [Vector Search and ANN](vector_search.md) | `vector_search` | Vector Search and ANN — HNSW, FAISS, ScaNN, approximate nearest neighbors at scale | vector-search, ann, hnsw, faiss |
| [Ranking Metrics and Evaluation](ranking_metrics.md) | `ranking_metrics` | Ranking Metrics and Evaluation — NDCG, MAP, MRR, precision at k, offline vs online evaluation | evaluation, ndcg, mrr, metrics |
| [Feature Engineering for RecSys](feature_engineering_recsys.md) | `feature_engineering_recsys` | Feature Engineering for RecSys — User and item features, interaction features, real-time computation | features, recsys, realtime |
| [Cold Start Problem](cold_start_problem.md) | `cold_start_problem` | Cold Start Problem — New user and item strategies, exploration vs exploitation | cold-start, exploration, onboarding |
| [Embeddings and Representation Learning](embeddings.md) | `embeddings` | Embeddings and Representation Learning — Dense vector representations for retrieval, similarity, and transfer | embeddings, representation-learning, similarity |
| [Transformer Architecture](transformer_architecture.md) | `transformer_architecture` | Transformer Architecture — Attention, self-attention, positional encoding, encoder-decoder | transformers, attention, architecture |
| [Deep Learning Fundamentals](deep_learning_fundamentals.md) | `deep_learning_fundamentals` | Deep Learning Fundamentals — Backprop, optimization, regularization, CNN, RNN architectures | deep-learning, optimization, cnn, rnn |
| [Classical ML Algorithms](classical_ml.md) | `classical_ml` | Classical ML Algorithms — Trees, SVMs, ensembles, XGBoost, LightGBM for production ranking | classical-ml, xgboost, lightgbm |
| [Experiment Design and A/B Testing](experiment_design.md) | `experiment_design` | Experiment Design and A/B Testing — Causal inference, statistical significance, online experiments | ab-testing, experimentation, causal-inference |
| [Loss Functions and Optimization](loss_functions_and_optimization.md) | `loss_functions_and_optimization` | Loss Functions and Optimization — Cross-entropy, contrastive loss, SGD variants, learning rate schedules | loss-functions, optimization, sgd |
| [LLM Fundamentals](llm_fundamentals.md) | `llm_fundamentals` | LLM Fundamentals — Architecture, pretraining, tokenization, scaling laws | llm, tokenization, pretraining |
| [Retrieval-Augmented Generation](rag_systems.md) | `rag_systems` | Retrieval-Augmented Generation — Combining search and retrieval with LLM generation | rag, retrieval, generation |
| [LLM Fine-Tuning](llm_fine_tuning.md) | `llm_fine_tuning` | LLM Fine-Tuning — LoRA, QLoRA, PEFT, parameter-efficient adaptation | fine-tuning, lora, peft |
| [LLM Evaluation](llm_evaluation.md) | `llm_evaluation` | LLM Evaluation — RAGAS, LLM-as-judge, production eval pipelines | llm-eval, ragas, eval-pipelines |
| [Prompt Engineering](prompt_engineering.md) | `prompt_engineering` | Prompt Engineering — Structured prompting, chain-of-thought, tool use patterns | prompting, tool-use, patterns |
| [ML System Design](ml_system_design.md) | `ml_system_design` | ML System Design — End-to-end design of production ML systems at scale | ml-system-design, architecture, production |
| [MLOps and ML Infrastructure](mlops.md) | `mlops` | MLOps and ML Infrastructure — CI/CD for ML, model registries, experiment tracking | mlops, cicd, model-registry |
| [Feature Stores](feature_stores.md) | `feature_stores` | Feature Stores — Online and offline serving, consistency, Feast and Tecton patterns | feature-store, online-serving, offline |
| [Model Serving and Inference](model_serving.md) | `model_serving` | Model Serving and Inference — TorchServe, Triton, latency optimization, batching strategies | model-serving, inference, triton, latency |
| [Data Pipelines](data_pipelines.md) | `data_pipelines` | Data Pipelines — Batch and streaming with Spark, Flink, Airflow, data quality | data-pipelines, spark, flink, airflow |
| [ML Monitoring and Observability](monitoring_and_observability.md) | `monitoring_and_observability` | ML Monitoring and Observability — Drift detection, model degradation, alerting | monitoring, observability, drift |
| [Python Engineering](python_engineering.md) | `python_engineering` | Python Engineering — Production Python, typing, async, packaging, testing | python, engineering, testing, typing |
| [TypeScript Engineering](typescript_engineering.md) | `typescript_engineering` | TypeScript Engineering — Full-stack capability, type safety, frontend and backend | typescript, full-stack, typing |
| [Distributed Systems](distributed_systems.md) | `distributed_systems` | Distributed Systems — Consensus, partitioning, CAP theorem, replication | distributed-systems, cap, consensus |
| [System Design Fundamentals](system_design_fundamentals.md) | `system_design_fundamentals` | System Design Fundamentals — Load balancing, caching, databases, API design | system-design, caching, databases |
| [Algorithms and Data Structures](algorithms_and_data_structures.md) | `algorithms_and_data_structures` | Algorithms and Data Structures — Core CS for interviews and production problem solving | algorithms, data-structures, cs |
| [Technical Leadership](technical_leadership.md) | `technical_leadership` | Technical Leadership — Cross-team influence, architectural ownership, mentoring | leadership, mentoring, architecture |
| [ML Strategy and Problem Framing](ml_strategy.md) | `ml_strategy` | ML Strategy and Problem Framing — When to use ML vs rules, scoping, measuring business impact | ml-strategy, product, impact |
| [Math Foundations for ML](math_foundations.md) | `math_foundations` | Math Foundations for ML — Linear algebra, probability, statistics, calculus | math, linear-algebra, probability, statistics |
| [Ethics and Responsible AI](ethics_and_responsible_ai.md) | `ethics_and_responsible_ai` | Ethics and Responsible AI — Bias, fairness, safety, and governance | responsible-ai, fairness, safety |

---

## Browse by Theme

### Ranking, Retrieval, and Recommenders
- [Learning to Rank](learning_to_rank.md) (`learning_to_rank`)
- [Multi-Stage Ranking Pipelines](multi_stage_ranking.md) (`multi_stage_ranking`)
- [Ranking Metrics and Evaluation](ranking_metrics.md) (`ranking_metrics`)
- [Search and Information Retrieval](search_and_retrieval.md) (`search_and_retrieval`)
- [Vector Search and ANN](vector_search.md) (`vector_search`)
- [Two-Tower Architecture](two_tower_architecture.md) (`two_tower_architecture`)
- [Embeddings and Representation Learning](embeddings.md) (`embeddings`)
- [Recommendation Systems](recommendation_systems.md) (`recommendation_systems`)
- [Feature Engineering for RecSys](feature_engineering_recsys.md) (`feature_engineering_recsys`)
- [Cold Start Problem](cold_start_problem.md) (`cold_start_problem`)

### Deep Learning and Modeling Foundations
- [Deep Learning Fundamentals](deep_learning_fundamentals.md) (`deep_learning_fundamentals`)
- [Transformer Architecture](transformer_architecture.md) (`transformer_architecture`)
- [Loss Functions and Optimization](loss_functions_and_optimization.md) (`loss_functions_and_optimization`)
- [Classical ML Algorithms](classical_ml.md) (`classical_ml`)
- [Math Foundations for ML](math_foundations.md) (`math_foundations`)

### LLMs and RAG
- [LLM Fundamentals](llm_fundamentals.md) (`llm_fundamentals`)
- [Prompt Engineering](prompt_engineering.md) (`prompt_engineering`)
- [Retrieval-Augmented Generation](rag_systems.md) (`rag_systems`)
