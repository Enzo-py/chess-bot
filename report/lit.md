## DeepChess: End-to-End Deep Neural Network for Automatic Learning in Chess (David, Netanyahu, Wolf)

- First end-to-end machine learning-based method achieving grandmaster-level chess without prior chess knowledge.
- two-phase training approach: (1) unsupervised deep belief network pretraining (Pos2Vec) to extract positional features, (2) supervised training of a Siamese network to compare positions
- uses position comparison approach instead of traditional scalar evaluation
- Training used millions of chess positions from real games
- Network distillation to compress model for faster inference
- Outperformed classic engines when given equal time constraints

#### Are we related?:
- We use modular approach - DeepChess uses a monolithic end-to-end architecture
- could try? to adapt their position comparison technique within RL framework

## Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (AlphaZero)

- Generalized the AlphaGo Zero approach to chess and shogi without domain-specific adaptations
- Combined Monte Carlo Tree Search (MCTS) with deep neural networks for evaluation
- Neural network outputs both policy (move probabilities) and value (position evaluation)
- Trained entirely through self-play reinforcement learning
- Used a general non-linear network to evaluate chess positions instead of handcrafted evaluation
- Achieved superhuman performance within 24 hours of training
- Demonstrated that a general-purpose MCTS with neural guidance outperforms specialized alpha-beta search algorithms

#### Are we related?:
- Both utilize reinforcement learning approaches for chess
- modular framework could be compared to AlphaZero's general algorithm approach
- new implementation? could incorporate the MCTS+neural network evaluation technique
- focus on limited computational resources addresses a key limitation of AlphaZero
- evaluation against Stockfish follows similar benchmarking methodology
- AlphaZero's self-play methodology could be incorporated in our training pipeline?

