from biobrain.salience.fingerprint import (
    Fingerprint, compute_fingerprint, RoleFingerprintIndex,
)
from biobrain.salience.roles import Role


class MockEntity:
    """Minimal entity stand-in for fingerprint tests."""
    def __init__(self, color, quadrant):
        self.color = color
        self.id = id(self)
        self._quadrant = quadrant
        class Region:
            cells = frozenset()
            area = 1
        self.region = Region()


def _quadrant_of(entity):
    return entity._quadrant


def test_fingerprint_three_granularities():
    """compute_fingerprint produces F_tight, F_mid, F_loose."""
    entities = [MockEntity(5, 2), MockEntity(7, 8)]
    role_assignments = {entities[0].id: Role.SELECTOR,
                         entities[1].id: Role.CURSOR}
    fp = compute_fingerprint(entities, role_assignments, _quadrant_of)
    assert isinstance(fp, Fingerprint)
    assert (Role.SELECTOR, 5, 2) in fp.tight
    assert (Role.CURSOR, 7, 8) in fp.tight
    assert (Role.SELECTOR, 5) in fp.mid
    assert Role.SELECTOR in fp.loose
    assert Role.CURSOR in fp.loose


def test_fingerprint_stability():
    """Same entities + roles -> same fingerprint."""
    entities = [MockEntity(5, 2), MockEntity(7, 8)]
    role_assignments = {entities[0].id: Role.SELECTOR,
                         entities[1].id: Role.CURSOR}
    fp1 = compute_fingerprint(entities, role_assignments, _quadrant_of)
    fp2 = compute_fingerprint(entities, role_assignments, _quadrant_of)
    assert fp1.tight == fp2.tight
    assert fp1.mid == fp2.mid
    assert fp1.loose == fp2.loose
    assert hash(fp1.mid) == hash(fp2.mid)


def test_fingerprint_color_invariance_in_mid():
    """Same role + color, different quadrant -> same F_mid, different F_tight."""
    e_a = MockEntity(5, 2)
    e_b = MockEntity(5, 8)
    roles_a = {e_a.id: Role.SELECTOR}
    roles_b = {e_b.id: Role.SELECTOR}
    fp_a = compute_fingerprint([e_a], roles_a, _quadrant_of)
    fp_b = compute_fingerprint([e_b], roles_b, _quadrant_of)
    assert fp_a.mid == fp_b.mid
    assert fp_a.tight != fp_b.tight


def test_fingerprint_index_insert_and_lookup_mid():
    """Insert subgoal under fingerprint; lookup by matching mid returns it."""
    idx = RoleFingerprintIndex()
    entities = [MockEntity(5, 2)]
    roles = {entities[0].id: Role.SELECTOR}
    fp = compute_fingerprint(entities, roles, _quadrant_of)
    idx.insert(fp, subgoal="subgoal_A")

    e2 = MockEntity(5, 8)
    fp2 = compute_fingerprint([e2], {e2.id: Role.SELECTOR}, _quadrant_of)
    results = idx.lookup(fp2)
    assert "subgoal_A" in results, "F_mid should match across quadrants"


def test_fingerprint_index_no_match_different_role():
    """Different role -> no match."""
    idx = RoleFingerprintIndex()
    e1 = MockEntity(5, 2)
    fp1 = compute_fingerprint([e1], {e1.id: Role.SELECTOR}, _quadrant_of)
    idx.insert(fp1, subgoal="A")

    e2 = MockEntity(5, 2)
    fp2 = compute_fingerprint([e2], {e2.id: Role.CURSOR}, _quadrant_of)
    assert idx.lookup(fp2) == []


def test_fingerprint_index_reset_clears():
    """reset_game wipes the index."""
    idx = RoleFingerprintIndex()
    e = MockEntity(5, 2)
    fp = compute_fingerprint([e], {e.id: Role.SELECTOR}, _quadrant_of)
    idx.insert(fp, "x")
    assert idx.lookup(fp) == ["x"]
    idx.reset_game()
    assert idx.lookup(fp) == []
