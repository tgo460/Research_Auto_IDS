from argparse import Namespace

from src_replica.train_improved_light_model import train_light_model


def test_train_light_model_starts_without_name_error(tmp_path):
    # Empty dataset dir should early-exit gracefully; this catches pre-fix NameError.
    args = Namespace(
        data_dir=str(tmp_path),
        output_dir=str(tmp_path / "models"),
        epochs=1,
        batch_size=2,
        lr=1e-3,
        max_rows=10,
        full_data=False,
    )
    train_light_model(args)
