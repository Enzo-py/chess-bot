
import traceback
from src.utils.socket_server import ServerSocket
import src.utils.message as protocol

from meta import AVAILABLE_MODELS

from src.chess.game import Game
from src.chess.player import Player

import asyncio
import json

class Server:
    """
    Server class that handles the app.
    """

    def __init__(self):
        self.server = ServerSocket(_print=True)

        self.focused_game = None

    async def run(self):

        await self.server.start()

        self.server.on(
            ServerSocket.EVENTS_TYPES.on_client_connect,
            "client-connect",
            lambda client: asyncio.create_task(self.server.send(client, protocol.Message("confirm-connection", "Connection established").to_json()))
        )

        self.server.on(
            ServerSocket.EVENTS_TYPES.on_message,
            "start-game",
            lambda client, message: self.start_game(message.content) if message.type == "start-game" else None
        )

        self.server.on(
            ServerSocket.EVENTS_TYPES.on_message,
            "get-possible-moves",
            lambda client, message: self.get_possible_moves(message.content) if message.type == "get-possible-moves" else None
        )

        self.server.on(
            ServerSocket.EVENTS_TYPES.on_message,
            "move-piece",
            lambda client, message: self.move_piece(message.content) if message.type == "move-piece" else None
        )

        self.server.on(
            ServerSocket.EVENTS_TYPES.on_message,
            "get-players-list",
            lambda client, message: self.get_players_list() if message.type == "get-players-list" else None
        )

        self.server.on(
            ServerSocket.EVENTS_TYPES.on_message,
            "create-player",
            lambda client, message: self.create_player(message.content) if message.type == "create-player" else None
        )

        # Main loop
        while self.server.running:
            await asyncio.sleep(2)

    def start_game(self, info):
        """
        Start a new game with the given info.
        """
        self.focused_game = Game()
        if info["game_mode"] == "PvP":
            if info["player_color"] == "w":
                self.focused_game.play(white=info["player1"], black=info["player2"])
            else:
                self.focused_game.play(white=info["player2"], black=info["player1"])
        elif info["game_mode"] == "PvAI":
            ai = AVAILABLE_MODELS[info["ai_selection"]]()
            if info["player_color"] == "w":
                self.focused_game.play(white=info["player"], black=ai)
            else:
                self.focused_game.play(white=ai, black=info["player"])

        elif info["game_mode"] == "AIvAI":
            ai1 = AVAILABLE_MODELS[info["ai1_selection"]]()
            ai2 = AVAILABLE_MODELS[info["ai2_selection"]]()
            if info["player_color"] == "w":
                self.focused_game.play(white=ai1, black=ai2)
            else:
                self.focused_game.play(white=ai2, black=ai1)

        self.focused_game.ia_move_handler = self.ia_move_handler
        ctn = {
            "FEN": self.focused_game.to_FEN(),
            "current_player": self.focused_game.current_player
        }
        asyncio.create_task(self.server.broadcast(protocol.Message("game-started", ctn).to_json()))
        self.focused_game.play_algo_move()

    def get_possible_moves(self, info):
        """
        Get the possible moves for the given position.
        """
        if self.focused_game is None:
            asyncio.create_task(self.server.broadcast(protocol.Message("error", "No game started").to_json()))
            return
        
        piece = self.focused_game.board.get(info["pos"], _exception=False)
        if piece is None:
            asyncio.create_task(self.server.broadcast(protocol.Message("error", "No piece at position").to_json()))
            return
        
        if piece.fen() != info["fen"]:
            asyncio.create_task(self.server.broadcast(protocol.Message("error", f"Invalid piece at position; find: {piece.fen()}, should be {info['fen']}").to_json()))
            return
        
        moves = piece.get_possible_moves(self.focused_game.board)
        moves += piece.get_possible_captures(self.focused_game.board)
        moves = list(set(moves))

        # transform coordinates to box names
        moves = [self.focused_game.board.get_box(m).upper() for m in moves]
        asyncio.create_task(self.server.broadcast(protocol.Message("possible-moves", {'moves': moves}).to_json()))

    def move_piece(self, info):
        """
        Move the piece from start to end.
        """
        if self.focused_game is None:
            asyncio.create_task(self.server.broadcast(protocol.Message("error", "No game started").to_json()))
            return
        
        try:
            self.focused_game.move(info["start"], info["end"])
        except Exception as e:
            asyncio.create_task(self.server.broadcast(protocol.Message("error", str(e)).to_json()))
            traceback.print_exc()

            # reset turn
            self.focused_game.current_player = Player.WHITE if self.focused_game.current_player == Player.BLACK else Player.BLACK
            return
        
        if info["promote"] is not None:
            self.focused_game.board.promote(info["end"], info["promote"])

        ctn = {
            "FEN": self.focused_game.to_FEN(),
            "king_in_check": self.focused_game.board.king_in_check['w'] or self.focused_game.board.king_in_check['b'],
            "checkmate": self.focused_game.board.checkmate
        }

        asyncio.create_task(self.server.broadcast(protocol.Message("confirm-move", ctn).to_json()))
        self.focused_game.play_algo_move()

    def ia_move_handler(self, action):
        """
        Handle the move of the AI.
        """
        ctn = {
            "FEN": self.focused_game.to_FEN(),
            "king_in_check": self.focused_game.board.king_in_check['w'] or self.focused_game.board.king_in_check['b'],
            "checkmate": self.focused_game.board.checkmate,
            "from": action["from"].upper(),
            "to": action["to"].upper(),
            "promote": action.get("promote")
        }
        asyncio.create_task(self.server.broadcast(protocol.Message("ai-move", ctn).to_json()))

    def get_players_list(self):
        """
        Get all registered players and AI.
        """
        ranking = json.load(open("./data/ranking.json", "r"))
        
        ais = []
        players = []
        for player in ranking:
            if player["type"] == "AI":
                if player["model"] not in AVAILABLE_MODELS: raise Exception(f"Model {player['model']} not found")

                ais.append(player)
            else:
                players.append(player)

        for model in AVAILABLE_MODELS:
            if model not in [ai["model"] for ai in ais]:
                ais.append({"model": model, "type": "AI", "elo": 600, "games": []})

        all_players = players + ais
        all_players = sorted(all_players, key=lambda x: x["elo"], reverse=True)
        json.dump(all_players, open("./data/ranking.json", "w"), indent=4)

        ctn = {
            "players": players,
            "ais": ais
        }
        asyncio.create_task(self.server.broadcast(protocol.Message("players-list", ctn).to_json()))

    def create_player(self, info):
        """
        Create a new player with the given name.
        """
        ranking = json.load(open("./data/ranking.json", "r"))
        if len(info['name']) < 3:
            asyncio.create_task(self.server.broadcast(protocol.Message("error", "Name too short").to_json()))
            return
        
        if info['name'] in [player["name"] for player in ranking if player["type"] == "player"]:
            asyncio.create_task(self.server.broadcast(protocol.Message("error", "Player already exists").to_json()))
            return
        
        ranking.append({
            "name": info['name'],
            "type": "player",
            "elo": 600,
            "games": []
        })
        json.dump(ranking, open("./data/ranking.json", "w"), indent=4)
        asyncio.create_task(self.server.broadcast(protocol.Message("player-created", "Player created").to_json()))

if __name__ == "__main__":
    Server = Server()
    asyncio.run(Server.run())

