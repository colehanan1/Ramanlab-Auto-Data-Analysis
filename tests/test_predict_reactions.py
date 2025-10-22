import pandas as pd

from fbpipe.steps.predict_reactions import _drop_flagged_flies, NON_REACTIVE_SPAN_PX


def test_drop_flagged_flies_removes_non_reactive_pairs():
    df = pd.DataFrame(
        {
            "dataset": ["set1", "set1", "set1", "set1"],
            "fly": ["flyA", "flyA", "flyB", "flyB"],
            "fly_number": ["1", "1", "2", "2"],
            "global_min": [10.0, 10.0, 15.0, 15.0],
            "global_max": [25.0, 25.0, 37.0, 60.0],
            "feature": [0.1, 0.2, 0.3, 0.4],
        }
    )

    filtered, flagged = _drop_flagged_flies(df, threshold=NON_REACTIVE_SPAN_PX)

    assert ("set1", "flyA", "1") in flagged
    assert ("set1", "flyB", "2") not in flagged
    assert len(filtered) == 2
    assert set(filtered["fly"].unique()) == {"flyB"}
    assert set(filtered["fly_number"].unique()) == {"2"}
