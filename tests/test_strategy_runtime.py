from __future__ import annotations

from sigmaevolve import strategy_runtime


def test_seed_everything_returns_cpu_when_cuda_unavailable(monkeypatch):
    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available():
                return False

            @staticmethod
            def manual_seed_all(seed):
                raise AssertionError(f"manual_seed_all should not be called: {seed}")

        @staticmethod
        def manual_seed(seed):
            return None

    monkeypatch.setitem(strategy_runtime.sys.modules, "torch", FakeTorch)

    assert strategy_runtime._seed_everything(1234) == "cpu"


def test_seed_everything_returns_cuda_when_available(monkeypatch):
    state = {"manual_seed": [], "manual_seed_all": []}

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def manual_seed_all(seed):
                state["manual_seed_all"].append(seed)

        @staticmethod
        def manual_seed(seed):
            state["manual_seed"].append(seed)

    monkeypatch.setitem(strategy_runtime.sys.modules, "torch", FakeTorch)

    assert strategy_runtime._seed_everything(7) == "cuda"
    assert state == {"manual_seed": [7], "manual_seed_all": [7]}
