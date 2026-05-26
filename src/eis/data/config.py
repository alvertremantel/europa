from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement

SPLITS = ("train", "val", "test")
BINARY_OPERATIONS = ("+", "-", "*", "/")
COMPARISON_OPERATIONS = ("<", ">")
COMPOSITE_OPERATIONS = ("+", "-", "*")
NUMBER_WIDTH = 6
VAL_SAMPLES_PER_KIND = 16
TEST_SAMPLES_PER_KIND = 16
SAMPLED_TRAIN_SAMPLES_PER_KIND = 128
SAMPLED_KIND_MAX_ATTEMPTS = 200_000


@dataclass(frozen=True)
class Band:
    name: str
    start: int
    end: int

    def values(self) -> range:
        return range(self.start, self.end + 1)


BANDS = (
    Band("small", 0, 20),
    Band("medium", 21, 100),
    Band("large", 101, 500),
)
_BANDS_BY_NAME = {band.name: band for band in BANDS}
_BAND_NAMES = tuple(band.name for band in BANDS)
_BAND_ORDER = {band.name: index for index, band in enumerate(BANDS)}
_TWO_BAND_PATTERNS = tuple(combinations_with_replacement(_BAND_NAMES, 2))
_THREE_BAND_PATTERNS = tuple(combinations_with_replacement(_BAND_NAMES, 3))


@dataclass(frozen=True)
class Config:
    seed: int = 42
    output_dir: str = "data"
    validate: bool = True
