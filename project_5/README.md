# MIT 6.86x — Project 5: Reinforcement Learning for a Text-Based Game

## Overview

This project implements and compares three reinforcement learning agents that learn to play a text-based "home adventure" game purely from natural-language state descriptions — no structured game state is given to the agent. The agent must learn, from text alone, which (action, object) command resolves its current quest, while coping with partial observability: the same underlying room is described using several different, randomly chosen sentences each time the agent visits it.

This is **Project 5** of [MIT 6.86x: Machine Learning with Python — From Linear Models to Deep Learning](https://www.edx.org/course/machine-learning-with-python-from-linear-models-to).

---

## The Game

The environment (`framework.py`) is a 4-room house:

| Room | Valid commands | Transitions to |
|---|---|---|
| Living | go south / go west / watch tv | Bedroom / Garden / Living |
| Garden | go south / go east / exercise bike | Kitchen / Living / Garden |
| Kitchen | go north / go east / eat apple | Garden / Bedroom / Kitchen |
| Bedroom | go north / go west / sleep bed | Living / Kitchen / Bedroom |

Each room is rendered to the agent as **one of several different text descriptions**, chosen at random on every visit (e.g., Kitchen might appear as "This room has a fridge, oven, and a sink" or "A red juicy fruit"). The hidden room identity is never given directly — the agent has to learn that differently-worded descriptions correspond to the same underlying state.

At the start of each episode, the agent is also given one of 4 random quests ("You are hungry," "You are sleepy," etc.), each resolved by exactly one (action, object) pair in exactly one room (e.g., "hungry" → `eat apple` in the Kitchen).

### Reward structure
- **+1** for executing the correct command to complete the active quest (terminal)
- **−0.01** per non-terminal step (encourages efficiency)
- **−0.1** additional penalty for an invalid command (e.g., `eat tv`)
- Episodes terminate after a quest is completed or **20 steps** are reached

---

## Project Structure

```
project_5/
├── framework.py         # Game environment: rooms, quests, transitions, reward logic
├── utils.py             # Bag-of-words feature extraction, data loading, EWMA smoothing
├── agent_tabular_ql.py  # Tabular Q-learning agent
├── agent_linear.py      # Linear Q-learning agent (BoW features)
├── agent_dqn.py         # Deep Q-learning agent (BoW features + neural net)
└── game.tsv             # Corpus of all room/quest text used to build the BoW vocabulary
```

---

## Three Agents, Three Levels of Generalization

The three agents solve the same game with progressively more generalizable state representations:

### 1. Tabular Q-Learning (`agent_tabular_ql.py`)
Maintains an explicit 4-D Q-table indexed by `[room_index, quest_index, action_index, object_index]`, with room/quest text descriptions mapped to discrete indices via dictionaries built from the known, finite set of all text variants.

- **Update rule:** `Q(s,c) ← (1−α)·Q(s,c) + α·(R + γ·max Q(s′,·))`
- **Limitation:** Each text variant is treated as a completely separate state — the agent cannot generalize what it learns about one room description to another description of the *same* room, and could not handle a description it hasn't seen during table construction.

### 2. Linear Q-Learning (`agent_linear.py`)
Replaces the lookup table with a weight matrix `θ` over a **bag-of-words vector representation** of the concatenated room + quest text (built in `utils.py`). Q-values are computed as `θ · state_vector` for each flattened (action, object) command index.

- **Update rule:** gradient step on the TD error, `θ[cmd_index] += α · (target − Q(s,c)) · state_vector`
- **Advantage over tabular:** because the state is represented by shared words rather than an opaque index, the agent generalizes across different phrasings of the same room — learning from one description's words transfers to another description that shares vocabulary.

### 3. Deep Q-Learning (`agent_dqn.py`)
Replaces the linear function with a small neural network (`torch.nn`) that takes the same bag-of-words vector and outputs **two separate Q-value heads** — one over actions, one over objects — trained jointly via squared TD-error loss (`torch.optim.SGD`).

- **Architecture:** `Linear(state_dim → hidden) → ReLU → [Linear(hidden → NUM_ACTIONS), Linear(hidden → NUM_OBJECTS)]`
- **Advantage over linear:** can learn non-linear interactions between words in the state description, rather than being restricted to a linear scoring function.

All three use **epsilon-greedy exploration** (ε=0.5 training, ε=0.05 testing) and a discounted reward objective (γ=0.5), averaged over multiple independent runs to reduce variance from random initialization and the game's stochastic room-description sampling.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3** | Core language |
| **NumPy** | Tabular Q-table, linear agent's weight updates |
| **PyTorch** (`torch.nn`, `torch.optim`, `torch.nn.functional`) | Deep Q-network construction and training |
| **Matplotlib** | Reward-per-epoch learning curves |
| **tqdm** | Training progress bars |

---

## How to Run

```bash
python agent_tabular_ql.py
python agent_linear.py
python agent_dqn.py
```

Each script runs multiple independent training runs (5–10, depending on agent), trains for a number of epochs (each epoch = 25 training episodes + 50 evaluation episodes), and plots the test-time average discounted reward per epoch, averaged across runs.

---

## Key Takeaways

- Designed and compared three reinforcement learning agents (tabular, linear function approximation, deep Q-network) against a shared text-based game environment with partial observability
- Implemented epsilon-greedy exploration, TD-error-based Q-value updates, and discounted cumulative reward evaluation from scratch
- Used bag-of-words feature extraction to let linear and deep agents generalize across semantically similar but lexically different state descriptions — something a tabular approach structurally cannot do
- Designed a multi-head output structure (separate Q-values for action and object) to decompose a combinatorial action space, rather than treating every (action, object) pair as an independent output
- Demonstrated the classic RL progression from tabular methods (exact but non-generalizing) to function approximation (generalizes, scales to larger/continuous state spaces)

---

## Course

This project was completed as part of **MIT 6.86x — Machine Learning with Python: From Linear Models to Deep Learning**.

It provides hands-on experience with reinforcement learning, Q-learning, function approximation, natural-language state representation, and epsilon-greedy exploration in a partially observable environment.