# Final Project: Advanced Machine learning and Autonomous Agents

## Project Structure

### Backend
- The chess game logic can be found in `backend/src/chess`.
- To create a model (AI), create a new file in `backend/models` and inherit the model from `Algo` (`backend/models/algo.py Algo`).
- To make a model available to the front interface, add it in `backend/meta.py AVAILABLE_MODELS`.
- The backend communic to the frontend using a socket server (code in `backend/src/utils/`)
- The server backend is available in `backend/server.py`.

### Frontend
- The frontend is a simple HTML/CSS/JS interface.
- `frontend/index.html` is the main page.
- `frontend/js/message.js` contains the code to read messages from the backend.
- You can find basic communication method in `index.html` such as `send_message` and `wait_for_message`.
- We are using d3.js to render the chess board.

## Getting Started

### Prerequisites
- Python 3.x
- Run `pip install -r requirements.txt` to install the required packages.
- To use stockfish, download the binary from [stockfishchess.org](https://stockfishchess.org/download/) and put the correct path in `backend/models/stockfish.py`.

### Running the Project
1. Run backend:
    - Open a terminal at the root of the project
    - execute: `cd backend`
    - execute: `python server.py`

2. Run the frontend: 
    - Open a terminal at the root of the projecvt
    - execute: `cd frontend`
    - execute: `python -m http.server`

3. Open a browser and go to `http://localhost:8000/`

## Notes
- We chose an HTML/CSS interface over Pygame because, although Pygame is compatible with notebooks, it is challenging to code complex features.
- The chess objects can provide ASCII art visualizations, but using the web interface is recommended.