from src.chess.game import Game
from src.chess.puzzle import Puzzle

from typing import Generator
import zstandard as zstd
import pandas as pd
import io

class Loader:

    def load(self, path: str, dtype: type = Game, chunksize: int = 128) -> Generator:
        """
        Load a .csv.zst or .pgn.zst file.

        :param path: path to the file
        :type path: str
        :param dtype: type of the data to load (Game or Puzzle)
        :type dtype: type
        :param chunksize: size of the chunks to load (only for CSV)
        :type chunksize: int
        :return: generator of data
        :rtype: Generator[list]
        """
        
        if dtype not in [Game, Puzzle]:
            raise Exception("Invalid dtype, must be Game or Puzzle")
        
        if path.endswith(".csv.zst"):
            yield from self._stream_csv_zst(path, dtype, chunksize)
        elif path.endswith(".pgn.zst"):
            yield from self._stream_pgn_zst(path, dtype, chunksize)
        else:
            raise Exception("Invalid file format, must be .csv.zst or .pgn.zst")

    def _stream_csv_zst(self, filepath, dtype, chunksize=128):
        """Stream a .csv.zst file in chunks."""
        
        with open(filepath, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            stream_reader = dctx.stream_reader(f)
            text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')

            for chunk in pd.read_csv(text_stream, chunksize=chunksize):
                data = [dtype().load(list(row[1].values)) for row in chunk.iterrows()]
                yield data

    def _stream_pgn_zst(self, filepath, dtype, chunksize=128):
        """Stream a .pgn.zst file, decompressing line by line."""
        
        with open(filepath, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            stream_reader = dctx.stream_reader(f)
            text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')

            buffer = ""
            games = []
            for line in text_stream:
                buffer += line
                if line.strip() == "":  # Empty line signals end of PGN game
                    game = dtype().load(buffer, format="pgn")
                    games.append(game)
                    buffer = ""  # Reset buffer for next game

                    if len(games) == chunksize:
                        yield games
                        games = []

            if buffer.strip():  # Handle last game if no trailing newline
                yield [dtype().load(buffer)]
