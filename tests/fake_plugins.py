"""Factories used by load_system tests ("fake_plugins:<factory>")."""

from fakes import FakeRAGSystem

NOT_A_FACTORY = 42


def make_good(**kwargs):
    return FakeRAGSystem(**kwargs)


class _NoAnswer:
    name = "no-answer"

    def retrieve(self, question, k):
        return []


def make_missing_answer():
    return _NoAnswer()


def make_raises():
    raise ValueError("bad configuration")
