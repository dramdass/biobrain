"""biobrain.perception — Spelke-grounded entity perception + Encoder boundary."""
from biobrain.perception.perceive import perceive, detect_events
from biobrain.perception.salience import Salience
from biobrain.perception.encoder import DefaultSpelkeEncoder, GRID_DIM
__all__ = ["perceive", "detect_events", "Salience",
           "DefaultSpelkeEncoder", "GRID_DIM"]
