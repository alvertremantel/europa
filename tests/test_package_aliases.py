from __future__ import annotations


def test_eur_ts_compatibility_attributes_resolve() -> None:
    import eur_ts
    import eur_ts.config
    import eur_ts.generator
    import eur_ts.trainer

    assert eur_ts.config is not None
    assert eur_ts.generator is not None
    assert eur_ts.trainer is not None


def test_eur_is_compatibility_attributes_resolve() -> None:
    import eur_is
    import eur_is.backend
    import eur_is.export

    assert eur_is.backend is not None
    assert eur_is.export is not None
