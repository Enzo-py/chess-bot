
import traceback
from src.utils.socket_server import ServerSocket
import src.utils.message as protocol

from meta import AVAILABLE_MODELS

from src.chess.game import Game
from src.chess.player import Player

import asyncio

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

        # Main loop
        while self.server.running:
            await asyncio.sleep(2)

    def start_game(self, info):
        """
        Start a new game with the given info.
        """
        self.focused_game = Game()
        if info["game_mode"] == "PvP":
            self.focused_game.play(white='player', black='player')
        elif info["game_mode"] == "PvAI":
            ai = AVAILABLE_MODELS[info["ai_selection"]]()
            if info["player_color"] == "white":
                self.focused_game.play(white='player', black=ai)
            else:
                self.focused_game.play(white=ai, black='player')

        elif info["game_mode"] == "AIvAI":
            ai1 = AVAILABLE_MODELS[info["ai1_selection"]]()
            ai2 = AVAILABLE_MODELS[info["ai2_selection"]]()
            if info["player_color"] == "white":
                self.focused_game.play(white=ai1, black=ai2)
            else:
                self.focused_game.play(white=ai2, black=ai1)

        ctn = {
            "FEN": self.focused_game.to_FEN(),
            "current_player": self.focused_game.current_player
        }
        asyncio.create_task(self.server.broadcast(protocol.Message("game-started", ctn).to_json()))

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
        
        if info["promote_to"] is not None:
            self.focused_game.board.promote(info["end"], info["promote_to"])

        ctn = {
            "FEN": self.focused_game.to_FEN(),
            "king_in_check": self.focused_game.board.king_in_check['w'] or self.focused_game.board.king_in_check['b'],
            "checkmate": self.focused_game.board.checkmate
        }

        asyncio.create_task(self.server.broadcast(protocol.Message("confirm-move", ctn).to_json()))

if __name__ == "__main__":
    Server = Server()
    asyncio.run(Server.run())

