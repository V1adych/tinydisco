"""ARC-AGI-3 games in JAX: games/base.py is the shared env class, games/<id>/ a game's logic, extractor and data;
originals/ (gitignored, filled by download.py) holds the Python originals the extractors and test_arc.py read."""

from .games.ls20.game import Ls20

ARC_GAMES = {"ls20": Ls20}
