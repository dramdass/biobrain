"""biobrain.env — ARC-AGI-3 environment coupling."""
try:
    from biobrain.env.arena_env import ArenaEnv
    __all__ = ["ArenaEnv"]
except ImportError:
    # arc_agi SDK not installed — biobrain core still usable for tests
    __all__ = []
