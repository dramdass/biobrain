# tests/test_roles.py
from biobrain.salience.roles import (
    Role, RoleSignature,
    ROLE_CATALOGUE, ROLE_DISCOVERY_K, role_likelihood,
    assign_role,
)


def test_role_catalogue_complete():
    """10 roles total, including UNKNOWN."""
    assert len(ROLE_CATALOGUE) == 10
    assert Role.UNKNOWN in ROLE_CATALOGUE
    expected = {Role.SELECTOR, Role.CURSOR, Role.PAINTER, Role.TARGET,
                Role.TOGGLE, Role.COUNTER, Role.BARRIER, Role.CONTAINER,
                Role.STATIC, Role.UNKNOWN}
    assert set(ROLE_CATALOGUE) == expected


def test_role_signature_initial():
    """Fresh signature has zero counters."""
    sig = RoleSignature()
    assert sig.n_observations == 0
    assert sig.clicked_caused_self_change == 0
    assert sig.clicked_caused_other_change == 0
    assert sig.translated_under_key_count == 0


def test_assign_role_below_threshold_returns_unknown():
    """Fewer than K=5 observations → UNKNOWN."""
    sig = RoleSignature(n_observations=3, clicked_caused_other_change=2)
    assert assign_role(sig) == Role.UNKNOWN


def test_assign_role_selector_signature():
    """High clicked_caused_other_change with low self_change → SELECTOR."""
    sig = RoleSignature(
        n_observations=10,
        clicked_on_count=5,
        clicked_caused_self_change=0,
        clicked_caused_other_change=4,
        clicked_caused_global_change=4,
        persistence=1.0,
    )
    assert assign_role(sig) == Role.SELECTOR


def test_assign_role_cursor_signature():
    """High translated_under_key_count dominant → CURSOR."""
    sig = RoleSignature(
        n_observations=15,
        clicked_on_count=0,
        translated_under_key_count=12,
        persistence=1.0,
    )
    assert assign_role(sig) == Role.CURSOR


def test_assign_role_target_signature():
    """High persistence + referenced_by_distance_goals → TARGET."""
    sig = RoleSignature(
        n_observations=20,
        clicked_on_count=0,
        translated_under_key_count=0,
        persistence=1.0,
        referenced_by_distance_goals=1,
    )
    assert assign_role(sig) == Role.TARGET


def test_assign_role_static_signature():
    """Low change rate everywhere, not referenced → STATIC."""
    sig = RoleSignature(
        n_observations=30,
        clicked_on_count=2,
        clicked_caused_self_change=0,
        clicked_caused_other_change=0,
        translated_under_key_count=0,
        persistence=1.0,
        referenced_by_distance_goals=0,
    )
    assert assign_role(sig) == Role.STATIC


def test_assign_role_barrier_signature():
    """Disappears on click + blocks → BARRIER."""
    sig = RoleSignature(
        n_observations=8,
        clicked_on_count=3,
        clicked_caused_self_change=3,
        persistence=0.4,  # disappears sometimes
        was_removed_on_click=2,
    )
    assert assign_role(sig) == Role.BARRIER


def test_role_likelihood_returns_dict():
    """role_likelihood returns posterior over all 10 roles."""
    sig = RoleSignature(n_observations=10, clicked_caused_other_change=5)
    likelihoods = role_likelihood(sig)
    assert set(likelihoods.keys()) == set(ROLE_CATALOGUE)
    assert all(0.0 <= v <= 1.0 for v in likelihoods.values())
    assert abs(sum(likelihoods.values()) - 1.0) < 1e-6


def test_role_likelihood_degenerate_collapses_to_unknown():
    """When n_observations crosses K but no causal signal is present,
    likelihood collapses to UNKNOWN (degenerate branch in role_likelihood).
    """
    # persistence=0 zeroes every role that depends on it (SELECTOR, PAINTER,
    # TARGET, TOGGLE, COUNTER, STATIC); no clicks/keys zero the rest.
    # UNKNOWN's score is also 0 at n_observations == K, so total == 0 and
    # the degenerate branch fires.
    sig = RoleSignature(n_observations=ROLE_DISCOVERY_K, persistence=0.0)
    likelihoods = role_likelihood(sig)
    assert likelihoods[Role.UNKNOWN] == 1.0
    assert all(v == 0.0 for r, v in likelihoods.items() if r != Role.UNKNOWN)


def test_container_is_stub_in_v0():
    """CONTAINER role currently has no signature (region-overlap tracking
    not yet built). Verify it never wins in any reasonable scenario.
    """
    # Build several signatures and confirm CONTAINER's score is always 0
    signatures = [
        RoleSignature(n_observations=10, clicked_on_count=5,
                       clicked_caused_other_change=3),
        RoleSignature(n_observations=20, translated_under_key_count=10),
        RoleSignature(n_observations=30, persistence=1.0),
    ]
    for sig in signatures:
        likelihoods = role_likelihood(sig)
        assert likelihoods[Role.CONTAINER] == 0.0


def test_selector_dominates_painter_when_no_self_change():
    """Locks the relative ordering: SELECTOR > PAINTER when clicks cause
    OTHER changes but not SELF changes (the cd82-selector signature).
    """
    sig = RoleSignature(
        n_observations=10,
        clicked_on_count=5,
        clicked_caused_self_change=0,
        clicked_caused_other_change=4,
        persistence=1.0,
    )
    likelihoods = role_likelihood(sig)
    assert likelihoods[Role.SELECTOR] > likelihoods[Role.PAINTER]
