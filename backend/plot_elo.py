import json
import matplotlib.pyplot as plt

# Load the ranking data
ranking_path = "data/stockfish_eval_results.json"
with open(ranking_path, "r") as f:
    ranking_data = json.load(f)

# Extract data for plotting
models = [entry['model'] for entry in ranking_data if 'model' in entry]
elos = [entry['elo'] for entry in ranking_data if 'elo' in entry]
games_played = [entry['nb_games'] for entry in ranking_data if 'nb_games' in entry]

# Plot ELO ratings
plt.figure(figsize=(10, 5))
plt.bar(models, elos, color='skyblue')
plt.xlabel('AI Models')
plt.ylabel('ELO Rating')
plt.title('ELO Ratings of AI Models')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot number of games played
plt.figure(figsize=(10, 5))
plt.bar(models, games_played, color='lightgreen')
plt.xlabel('AI Models')
plt.ylabel('Number of Games Played')
plt.title('Number of Games Played by AI Models')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show() 