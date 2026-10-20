"""wdd: write-direction dictionaries, training-free. The floor, the native-row dictionary, the exact ledger of a neuron's inputs, the K-input simulation, the weak and strong fields, and read energy. Every function is the one the record's experiments ran (docs/atlas, areas 62 to 68)."""
__version__ = "0.1.0"
from .model import load, MODELS, DEV, Capture, run_base, logprobs_at, kl, write_w, read_w, ln_apply, med
from .corpus import wikitext_ids, pile_ids, windows_from
from .floor import unitr, rotate, rows_dictionary, floor_calibration, floor_of, omp, omp_metric
from .ledger import Ledger
from .fields import field_directions, union, crowd_on_field
from .readenergy import sampled_fisher, read_energy, grade, blended_sqrt
