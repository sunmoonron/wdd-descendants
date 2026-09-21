import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
name = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else name
rev = os.environ.get("WDD_REV"); rnd = os.environ.get("WDD_RANDOM") == "1"; corpus = os.environ.get("WDD_CORPUS", "wikitext")
build_cache(name, tag=tag, revision=rev, random_init=rnd, corpus=corpus)
