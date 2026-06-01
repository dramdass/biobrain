from biobrain.salience.subgoals import (
    Subgoal, SubgoalDetector,
)
from biobrain.salience.fingerprint import Fingerprint
from biobrain.salience.roles import Role


def make_fp(mid_tuples):
    """Helper: build a Fingerprint with the given (role, color) mid set."""
    return Fingerprint(
        tight=frozenset(),
        mid=frozenset(mid_tuples),
        loose=frozenset(role for role, _ in mid_tuples),
    )


def test_subgoal_dataclass():
    """Subgoal stores start/end fingerprints, action subsequence,
    validation flag, source metadata.
    """
    sg = Subgoal(
        start_fp=make_fp([(Role.SELECTOR, 5)]),
        action_subsequence=(("click", 30, 4),),
        end_fp=make_fp([(Role.SELECTOR, 5), (Role.PAINTER, 0)]),
        critic_validated=False,
        source_level=0,
        source_attempt_id=1,
    )
    assert sg.start_fp != sg.end_fp
    assert len(sg.action_subsequence) == 1
    assert sg.critic_validated is False


def test_detector_no_change_no_subgoal():
    """If fingerprint doesn't change, no subgoal is detected."""
    det = SubgoalDetector()
    fp = make_fp([(Role.STATIC, 5)])
    sg = det.observe_transition(
        fingerprint_before=fp,
        fingerprint_after=fp,
        action=("noop",),
        critic_distance_dropped=False,
        source_level=0,
        source_attempt_id=0,
    )
    assert sg is None


def test_detector_fingerprint_change_creates_subgoal():
    """When F_mid changes, a Subgoal is returned."""
    det = SubgoalDetector()
    fp_a = make_fp([(Role.SELECTOR, 5)])
    fp_b = make_fp([(Role.SELECTOR, 5), (Role.PAINTER, 0)])
    sg = det.observe_transition(
        fingerprint_before=fp_a,
        fingerprint_after=fp_b,
        action=("click", 43, 4),
        critic_distance_dropped=False,
        source_level=0,
        source_attempt_id=1,
    )
    assert sg is not None
    assert sg.start_fp.mid == fp_a.mid
    assert sg.end_fp.mid == fp_b.mid
    assert sg.action_subsequence == (("click", 43, 4),)
    assert sg.critic_validated is False
    assert sg.source_level == 0


def test_detector_critic_validation():
    """critic_distance_dropped=True sets validated flag."""
    det = SubgoalDetector()
    fp_a = make_fp([(Role.SELECTOR, 5)])
    fp_b = make_fp([(Role.SELECTOR, 5), (Role.PAINTER, 0)])
    sg = det.observe_transition(
        fingerprint_before=fp_a,
        fingerprint_after=fp_b,
        action=("click", 43, 4),
        critic_distance_dropped=True,
        source_level=0,
        source_attempt_id=2,
    )
    assert sg is not None
    assert sg.critic_validated is True


def test_detector_accumulates_action_subsequence():
    """Actions between subgoals are accumulated into the next subgoal's
    subsequence.
    """
    det = SubgoalDetector()
    fp_a = make_fp([(Role.SELECTOR, 5)])
    fp_b = make_fp([(Role.SELECTOR, 5), (Role.PAINTER, 0)])
    det.observe_transition(fp_a, fp_a, ("key", 1), False, 0, 0)
    det.observe_transition(fp_a, fp_a, ("key", 2), False, 0, 0)
    sg = det.observe_transition(fp_a, fp_b, ("click", 43, 4), True, 0, 0)
    assert sg is not None
    assert sg.action_subsequence == (("key", 1), ("key", 2), ("click", 43, 4))


def test_detector_reset_attempt_clears_accumulator():
    """reset_attempt clears the pending action accumulator."""
    det = SubgoalDetector()
    fp_a = make_fp([(Role.SELECTOR, 5)])
    fp_b = make_fp([(Role.PAINTER, 0)])
    det.observe_transition(fp_a, fp_a, ("key", 1), False, 0, 0)
    det.reset_attempt()
    sg = det.observe_transition(fp_a, fp_b, ("click", 30, 4), False, 0, 1)
    assert sg.action_subsequence == (("click", 30, 4),)


def test_detector_reset_game_clears_accumulator():
    """reset_game also clears the accumulator."""
    det = SubgoalDetector()
    fp_a = make_fp([(Role.SELECTOR, 5)])
    fp_b = make_fp([(Role.PAINTER, 0)])
    det.observe_transition(fp_a, fp_a, ("key", 1), False, 0, 0)
    det.reset_game()
    sg = det.observe_transition(fp_a, fp_b, ("click", 30, 4), False, 0, 0)
    assert sg.action_subsequence == (("click", 30, 4),)
