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

## Checkmating One, by Using Many – Combining Mixture of Experts with MCTS to Improve in Chess

- integrates Mixture of Experts (MoE) with MCTS for chess.
- Uses specialized neural networks for different game phases (opening, middlegame, endgame).
- Demonstrates improved computational efficiency by selectively activating specific networks based on board position.
- Gains ~120 Elo points over a monolithic model like AlphaZero.
- Uses phase-based training strategies: separated learning, staged learning, and weighted learning.

#### Are we related?:
- We will incorporate the MoE concept into our modular RL framework,  specialized models for different chess phases.
- findings suggest using separate models per game phase improves efficiency, balances computational cost and playing strength.
- perhaps? can explore different training strategies (separated vs. staged learning) for our agents.
- we can compare a single RL model vs. a modular phase-based RL model, inspired by their methodology.

## Reinforcement Learning in an Adaptable Chess Environment for Detecting Human-understandable Concepts"

- many fast, lightweight chess environment with customizable board sizes (4x5, 6x6)
- implements self-play reinforcement learning agents with limited computational resources
- concept detection methods to analyze what concepts the agents internalize during training
- wow! open-source code for environment, trained agents, and concept probing tools
- shows agents learn key chess concepts like material advantage, mate threats, and opponent piece threats
- concept representation evolves throughout training and varies by network architecture
- explainable AI techniques to understand neural networks' internal representations

#### Are we related?:
- we only have the traditional chess board environment, don't have time for custom environments
- concept detection methods could help analyze what your chessbots are learning? seems difficult
- analysis of how different network architectures (CNN vs ResNet) affect learning, also our CNN is much smaller

## Q-Learning Algorithms: A Comprehensive Classification and Applications
Technical Overview and Main Contribution:

- comprehensive classification of Q-learning algorithms into single-agent and multi-agent approaches
- mathematical foundations of Q-learning, including MDP, value functions, and Bellman equations
-  evolution from basic Q-learning to advanced variants like Deep Q-learning
- specific algorithms including Hierarchical Q-learning, Double Q-learning, Modular Q-learning
- limitations of basic Q-learning (memory issues with large state spaces, overestimation bias)

#### Are we related?:
- this paper is very foundational, not directly related to chess
- Deep Q-learning with CNNs described in this paper could be implemented 
- memory limitations in basic Q-learning justifies modular approach
- hierarchical Q-learning might be useful for decomposing the complex chess problem

#### TODO: Find and read more papers