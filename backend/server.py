
from src.utils.socket_server import ServerSocket
import src.utils.message as protocol

from meta import AVAILABLE_MODELS

from src.chess.game import Game
from src.chess.player import Player
from models.engine import Engine

import traceback
import asyncio
import json
import chess

PATHS = {
    "ranking": ["./data/ranking.json", "backend/data/ranking.json"]
}

class Server:
    """
    Server class that handles the app.
    """

    def __init__(self):
        self.server = ServerSocket(_print=True)

        self.focused_game = None

        for model in AVAILABLE_MODELS.values():
            try: getattr(model, "__author__")
            except: raise Engine.UndefinedAuthorError(model, f"Model {model.__name__} has no author")

            try: getattr(model, "__description__")
            except: raise Engine.UndefinedDescriptionError(model, f"Model {model.__name__} has no description")

            try: getattr(model, "play")
            except: raise Engine.UndefinedPlayMethodError(model, f"Model {model.__name__} has no play method")

    def open_file(self, name, mode):
        """
        Open a file with the given name.

        Was create because some computer need different paths to access the same file. 
        (maybe it's because the person didn't follow the README)
        """
        if name in PATHS:
            for path in PATHS[name]:
                try:
                    return open(path, mode)
                except FileNotFoundError:
                    continue
                except Exception as e:
                    raise e
            raise Exception(f"File {name} not found")
        else:
            raise Exception(f"Path {name} not registered in PATHS")

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

    async def start_game(self, info):
        """
        Start a new game with the given info.
        """
        self.focused_game = Game()
        if info["game_mode"] == "PvP":
            if info["player_color"] == "w":
                self.focused_game.play(white=Player(info["player1"]), black=Player(info["player2"]))
            else:
                self.focused_game.play(white=Player(info["player2"]), black=Player(info["player1"]))
        elif info["game_mode"] == "PvAI":
            ai = AVAILABLE_MODELS[info["ai_selection"]]()
            if info["player_color"] == "w":
                self.focused_game.play(white=Player(info["player"]), black=ai)
            else:
                self.focused_game.play(white=ai, black=Player(info["player"]))

        elif info["game_mode"] == "AIvAI":
            ai1 = AVAILABLE_MODELS[info["ai1_selection"]]()
            ai2 = AVAILABLE_MODELS[info["ai2_selection"]]()
            if info["player_color"] == "w":
                self.focused_game.play(white=ai1, black=ai2)
            else:
                self.focused_game.play(white=ai2, black=ai1)

        self.focused_game.ia_move_handler = self.ia_move_handler
        ctn = {
            "FEN": self.focused_game.fen(),
            "current_player": self.focused_game.board.turn
        }
        asyncio.create_task(self.server.broadcast(protocol.Message("game-started", ctn).to_json()))

        # wait 1s before playing the first move
        await asyncio.sleep(0.8)
        self.focused_game.play_engine_move()

    def get_possible_moves(self, info):
        """
        Get the possible moves for the given position.
        """
        if self.focused_game is None:
            asyncio.create_task(self.server.broadcast(protocol.Message("error", "No game started").to_json()))
            return
        
        piece = self.focused_game.get_piece(info["pos"])
        if piece is None:
            asyncio.create_task(self.server.broadcast(protocol.Message("error", "No piece at position").to_json()))
            return
        
        if str(piece) != info["fen"]:
            asyncio.create_task(self.server.broadcast(protocol.Message("error", f"Invalid piece at position; find: {piece.fen()}, should be {info['fen']}").to_json()))
            return
        
        moves = self.focused_game.get_possible_moves(info["pos"])
        # transform coordinates to end box names
        moves = [chess.square_name(move.to_square).upper() for move in moves]
        asyncio.create_task(self.server.broadcast(protocol.Message("possible-moves", {'moves': moves}).to_json()))

    def move_piece(self, info):
        """
        Move the piece from start to end.
        """
        if self.focused_game is None:
            asyncio.create_task(self.server.broadcast(protocol.Message("error", "No game started").to_json()))
            return
        
        try:
            move = chess.Move.from_uci(info["start"].lower() + info["end"].lower() + (info.get("promote", "") or "").lower())
            self.focused_game.move(move)
        except Exception as e:
            asyncio.create_task(self.server.broadcast(protocol.Message("error", str(e)).to_json()))
            traceback.print_exc()
            return

        ctn = {
            "FEN": self.focused_game.fen(),
            "king_in_check": self.focused_game.king_in_check[chess.WHITE] or self.focused_game.king_in_check[chess.BLACK],
            "checkmate": "w" if self.focused_game.checkmate == chess.WHITE else "b" if self.focused_game.checkmate == chess.BLACK else None,
            "draw": self.focused_game.draw,
        }

        asyncio.create_task(self.server.broadcast(protocol.Message("confirm-move", ctn).to_json()))
        async def play():
            self.focused_game.play_engine_move()
        asyncio.create_task(play())

    def ia_move_handler(self, move: chess.Move):
        """
        Handle the move of the AI.
        """
        _from = chess.square_name(move.from_square).upper()
        _to = chess.square_name(move.to_square).upper()
        if move.promotion is not None:
            promote = chess.piece_symbol(move.promotion)
            # upper if white, lower if black (reverse in the if because the turn is already changed)
            promote = promote.upper() if self.focused_game.board.turn == chess.BLACK else promote.lower()
        else:
            promote = None

        ctn = {
            "FEN": self.focused_game.fen(),
            "king_in_check": self.focused_game.king_in_check[chess.WHITE] or self.focused_game.king_in_check[chess.BLACK],
            "checkmate": "w" if self.focused_game.checkmate == chess.WHITE else "b" if self.focused_game.checkmate == chess.BLACK else None,
            "draw": self.focused_game.draw,
            "from": _from,
            "to": _to,
            "promote": promote
        }
        asyncio.create_task(self.server.broadcast(protocol.Message("ai-move", ctn).to_json()))
        
        async def play():
            await asyncio.sleep(0.8)
            self.focused_game.play_engine_move()
        asyncio.create_task(play())

    def get_players_list(self):
        """
        Get all registered players and AI.
        """
        ranking = json.load(self.open_file("ranking", "r"))
        
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
        json.dump(all_players, self.open_file("ranking", "w"), indent=4)

        ctn = {
            "players": players,
            "ais": ais
        }
        asyncio.create_task(self.server.broadcast(protocol.Message("players-list", ctn).to_json()))

    def create_player(self, info):
        """
        Create a new player with the given name.
        """
        ranking = json.load(self.open_file("ranking", "r"))
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
        json.dump(ranking, self.open_file("ranking", "w"), indent=4)
        asyncio.create_task(self.server.broadcast(protocol.Message("player-created", "Player created").to_json()))

if __name__ == "__main__":
    Server = Server()
    asyncio.run(Server.run())

