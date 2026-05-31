"""biobrain.adapters — per-environment glue.

An adapter implements the `biobrain.protocols.Adapter` protocol: it
supplies an Encoder, env bindings, and optionally an initial affordance
prior. Brain library does not import adapter code.

The ARC-AGI-3 adapter (`biobrain.adapters.arc`) is the one we ship.
Future adapters (synthetic test env, alternate envs) live as siblings.
"""

__all__ = []
