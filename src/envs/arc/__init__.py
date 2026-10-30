"""ARC-AGI-3 games in JAX: games/base.py is the shared env class, games/<id>/ a game's logic, extractor and data;
originals/ (gitignored, filled by download.py) holds the Python originals the extractors and test_arc.py read.

Official games (ARC Prize Foundation, the 4 keyboard-only ones of the 25 public games; the rest need the click action):
  ls20  maze; carry a symbol whose shape/color/rotation tiles change, open locks that want the exact symbol; energy, lives, fog (7 levels)
  tr87  rewriting puzzle; edit an answer row with a cursor until it is the source row rewritten by the rules shown (6 levels)
  wa30  warehouse; grab/drag crates into goal zones while collector bots help and thief bots steal, step budget (9 levels, 5 actions)
  g50t  maze with plates, doors, teleporters, patrols; action 5 rewinds your run into a ghost that replays it beside you (7 levels)
Community games (github.com/theredbluepill/arc-interactive, MIT): one mechanic each on a small grid scaled to the frame, 5 levels
(gr01 6). Train set, 8:
  ul01 key opens door | sk01 sokoban | fs01 step on all plates, then exit | tp01 paired teleporters | ic01 slide until blocked
  gr01 gravity after each move | zq01 hazards blink on a period | bp01 power towers with action 5, then exit
Spare, 8 (ported, in neither set):
  va01 visit every cell | rs01 collect only the signposted color | ms01 hidden minefield with counts | nw01 arrow tiles force the next move
  co01 pads recolor you, matching doors open | sn01 snake | hs01 chased by a hunter (it never catches) | sv01 hunger and warmth meters, survive 60 steps
Holdout set, 8 (rules absent from training, built from familiar pieces):
  wk01 floor collapses behind you | bd01 entering a cell twice loses | or01 keys orbit a pillar | rv01 collect while sparks drift (harmless)
  dl01 moves run one action late, action 5 drops the queue | ob01 three avatars, action 5 picks one | hn01 tower of Hanoi | mb01 crates slide toward you
All games are deterministic; the frame is (64, 64) color indices; reward is +1 per level, done on win or loss."""

from .games.bd01.game import Bd01
from .games.bp01.game import Bp01
from .games.co01.game import Co01
from .games.dl01.game import Dl01
from .games.fs01.game import Fs01
from .games.g50t.game import G50t
from .games.gr01.game import Gr01
from .games.hn01.game import Hn01
from .games.hs01.game import Hs01
from .games.ic01.game import Ic01
from .games.ls20.game import Ls20
from .games.mb01.game import Mb01
from .games.ms01.game import Ms01
from .games.nw01.game import Nw01
from .games.ob01.game import Ob01
from .games.or01.game import Or01
from .games.rs01.game import Rs01
from .games.rv01.game import Rv01
from .games.sk01.game import Sk01
from .games.sn01.game import Sn01
from .games.sv01.game import Sv01
from .games.tp01.game import Tp01
from .games.tr87.game import Tr87
from .games.ul01.game import Ul01
from .games.va01.game import Va01
from .games.wa30.game import Wa30
from .games.wk01.game import Wk01
from .games.zq01.game import Zq01

OFFICIAL = ("ls20", "tr87", "wa30", "g50t")
TRAIN = ("ul01", "sk01", "fs01", "tp01", "ic01", "gr01", "zq01", "bp01")
HOLDOUT = ("wk01", "bd01", "or01", "rv01", "dl01", "ob01", "hn01", "mb01")
ARC_GAMES = {"bd01": Bd01, "bp01": Bp01, "co01": Co01, "dl01": Dl01, "fs01": Fs01, "g50t": G50t, "gr01": Gr01, "hn01": Hn01, "hs01": Hs01, "ic01": Ic01, "ls20": Ls20, "mb01": Mb01, "ms01": Ms01, "nw01": Nw01, "ob01": Ob01, "or01": Or01, "rs01": Rs01, "rv01": Rv01, "sk01": Sk01, "sn01": Sn01, "sv01": Sv01, "tp01": Tp01, "tr87": Tr87, "ul01": Ul01, "va01": Va01, "wa30": Wa30, "wk01": Wk01, "zq01": Zq01}
