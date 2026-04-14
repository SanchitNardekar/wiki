```markdown
# Wiki Index

A curated knowledge base spanning ranking/recsys, LLM systems, ML engineering, and core software/system design.

---

## Summary Table

| Page | Key | One-line summary |
|---|---|---|
| Learning to Rank | `learning_to_rank` | Learning to Rank — LambdaMART, RankNet, pointwise/pairwise/listwise approaches |
| Recommendation Systems | `recommendation_systems` | Recommendation Systems — Collaborative filtering, content-based, hybrid approaches |
| Two-Tower Architecture | `two_tower_architecture` | Two-Tower Architecture — Industry standard dual-encoder for retrieval and pre-ranking |
| Multi-Stage Ranking Pipelines | `multi_stage_ranking` | Multi-Stage Ranking Pipelines — Retrieval to L1 to L2 to re-ranking funnel and latency budgets |
| Search and Information Retrieval | `search_and_retrieval` | Search and Information Retrieval — BM25, TF-IDF, inverted indexes, query understanding |
| Vector Search and ANN | `vector_search` | Vector Search and ANN — HNSW, FAISS, ScaNN, approximate nearest neighbors at scale |
| Ranking Metrics and Evaluation | `ranking_metrics` | Ranking Metrics and Evaluation — NDCG, MAP, MRR, precision at k, offline vs online evaluation |
| Feature Engineering for RecSys | `feature_engineering_recsys` | Feature Engineering for RecSys — User and item features, interaction features, real-time computation |
| Cold Start Problem | `cold_start_problem` | Cold Start Problem — New user and item strategies, exploration vs exploitation |
| Embeddings and Representation Learning | `embeddings` | Embeddings and Representation Learning — Dense vector representations for retrieval, similarity, and transfer |
| Transformer Architecture | `transformer_architecture` | Transformer Architecture — Attention, self-attention, positional encoding, encoder-decoder |
| Deep Learning Fundamentals | `deep_learning_fundamentals` | Deep Learning Fundamentals — Backprop, optimization, regularization, CNN, RNN architectures |
| Classical ML Algorithms | `classical_ml` | Classical ML Algorithms — Trees, SVMs, ensembles, XGBoost, LightGBM for production ranking |
| Experiment Design and A/B Testing | `experiment_design` | Experiment Design and A/B Testing — Causal inference, statistical significance, online experiments |
| Loss Functions and Optimization | `loss_functions_and_optimization` | Loss Functions and Optimization — Cross-entropy, contrastive loss, SGD variants, learning rate schedules |
| LLM Fundamentals | `llm_fundamentals` | LLM Fundamentals — Architecture, pretraining, tokenization, scaling laws |
| Retrieval-Augmented Generation | `rag_systems` | Retrieval-Augmented Generation — Combining search and retrieval with LLM generation |
| LLM Fine-Tuning | `llm_fine_tuning` | LLM Fine-Tuning — LoRA, QLoRA, PEFT, parameter-efficient adaptation |
| LLM Evaluation | `llm_evaluation` | LLM Evaluation — RAGAS, LLM-as-judge, production eval pipelines |
| Prompt Engineering | `prompt_engineering` | Prompt Engineering — Structured prompting, chain-of-thought, tool use patterns |
| ML System Design | `ml_system_design` | ML System Design — End-to-end design of production ML systems at scale |
| MLOps and ML Infrastructure | `mlops` | MLOps and ML Infrastructure — CI/CD for ML, model registries, experiment tracking |
| Feature Stores | `feature_stores` | Feature Stores — Online and offline serving, consistency, Feast and Tecton patterns |
| Model Serving and Inference | `model_serving` | Model Serving and Inference — TorchServe, Triton, latency optimization, batching strategies |
| Data Pipelines | `data_pipelines` | Data Pipelines — Batch and streaming with Spark, Flink, Airflow, data quality |
| ML Monitoring and Observability | `monitoring_and_observability` | ML Monitoring and Observability — Drift detection, model degradation, alerting |
| Python Engineering | `python_engineering` | Python Engineering — Production Python, typing, async, packaging, testing |
| TypeScript Engineering | `typescript_engineering` | TypeScript Engineering — Full-stack capability, type safety, frontend and backend |
| Distributed Systems | `distributed_systems` | Distributed Systems — Consensus, partitioning, CAP theorem, replication |
| System Design Fundamentals | `system_design_fundamentals` | System Design Fundamentals — Load balancing, caching, databases, API design |
| Algorithms and Data Structures | `algorithms_and_data_structures` | Algorithms and Data Structures — Core CS for interviews and production problem solving |
| Technical Leadership | `technical_leadership` | Technical Leadership — Cross-team influence, architectural ownership, mentoring |
| ML Strategy and Problem Framing | `ml_strategy` | ML Strategy and Problem Framing — When to use ML vs rules, scoping, measuring business impact |
| Math Foundations for ML | `math_foundations` | Math Foundations for ML — Linear algebra, probability, statistics, calculus |
| Ethics and Responsible AI | `ethics_and_responsible_ai` | Ethics and Responsible AI — Bias, fairness, safety, and governance |

---

## Pages (by domain)

### Ranking, Search, and Recommendation
- **Learning to Rank** (`learning_to_rank`) — LambdaMART, RankNet, pointwise/pairwise/listwise approaches  
- **Recommendation Systems** (`recommendation_systems`) — Collaborative filtering, content-based, hybrid approaches  
- **Two-Tower Architecture** (`two_tower_architecture`) — Industry standard dual-encoder for retrieval and pre-ranking  
- **Multi-Stage Ranking Pipelines** (`multi_stage_ranking`) — Retrieval to L1 to L2 to re-ranking funnel and latency budgets  
- **Search and Information Retrieval** (`search_and_retrieval`) — BM25, TF-IDF, inverted indexes, query understanding  
- **Vector Search and ANN** (`vector_search`) — HNSW, FAISS, ScaNN, approximate nearest neighbors at scale  
- **Ranking Metrics and Evaluation** (`ranking_metrics`) — NDCG, MAP, MRR, precision at k, offline vs online evaluation  
- **Feature Engineering for RecSys** (`feature_engineering_recsys`) — User and item features, interaction features, real-time computation  
- **Cold Start Problem** (`cold_start_problem`) — New user and item strategies, exploration vs exploitation  
- **Embeddings and Representation Learning** (`embeddings`) — Dense vector representations for retrieval, similarity, and transfer  

### Deep Learning and LLMs
- **Transformer Architecture** (`transformer_architecture`) — Attention, self-attention, positional encoding, encoder-decoder  
- **Deep Learning Fundamentals** (`deep_learning_fundamentals`) — Backprop, optimization, regularization, CNN, RNN architectures  
- **Loss Functions and Optimization** (`loss_functions_and_optimization`) — Cross-entropy, contrastive loss, SGD variants, learning rate schedules  
- **LLM Fundamentals** (`llm_fundamentals`) — Architecture, pretraining, tokenization, scaling laws  
- **Retrieval-Augmented Generation** (`rag_systems`) — Combining search and retrieval with LLM generation  
- **LLM Fine-Tuning** (`llm_fine_tuning`) — LoRA, QLoRA, PEFT, parameter-efficient adaptation  
- **LLM Evaluation** (`llm_evaluation`) — RAGAS, LLM-as-judge, production eval pipelines  
- **Prompt Engineering** (`prompt_engineering`) — Structured prompting, chain-of-thought, tool use patterns  

### ML Engineering, Systems, and Operations
- **ML System Design** (`ml_system_design`) — End-to-end design of production ML systems at scale  
- **MLOps and ML Infrastructure** (`mlops`) — CI/CD for ML, model registries, experiment tracking  
- **Feature Stores** (`feature_stores`) — Online and offline serving, consistency, Feast and Tecton patterns  
- **Model Serving and Inference** (`model_serving`) — TorchServe, Triton, latency optimization, batching strategies  
- **Data Pipelines** (`data_pipelines`) — Batch and streaming with Spark, Flink, Airflow, data quality  
- **ML Monitoring and Observability** (`monitoring_and_observability`) — Drift detection, model degradation, alerting  
- **Experiment Design and A/B Testing** (`experiment_design`) — Causal inference, statistical significance, online experiments  

### ML and CS Fundamentals
- **Classical ML Algorithms** (`classical_ml`) — Trees, SVMs, ensembles, XGBoost, LightGBM for production ranking  
- **Math Foundations for ML** (`math_foundations`) — Linear algebra, probability, statistics, calculus  
- **Algorithms and Data Structures** (`algorithms_and_data_structures`) — Core CS for interviews and production problem solving  

### Software Engineering and System Design
- **Python Engineering** (`python_engineering`) — Production Python, typing, async, packaging, testing  
- **TypeScript Engineering** (`typescript_engineering`) — Full-stack capability, type safety, frontend and backend  
- **Distributed Systems** (`distributed_systems`) — Consensus, partitioning, CAP theorem, replication  
- **System Design Fundamentals** (`system_design_fundamentals`) — Load balancing, caching, databases, API design  

### Leadership, Strategy, and Responsible AI
- **Technical Leadership** (`technical_leadership`) — Cross-team influence, architectural ownership, mentoring  
- **ML Strategy and Problem Framing** (`ml_strategy`) — When to use ML vs rules, scoping, measuring business impact  
- **Ethics and Responsible AI** (`ethics_and_responsible_ai`) — Bias, fairness, safety, and governance  
```