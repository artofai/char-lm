import pandas as pd
import logging
import click
from pathlib import Path
from typing import List
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def preprocess_dataset(df: pd.DataFrame) -> List[str]:
    df = df.drop_duplicates("city")
    logger.info(f"Rows after deduplication: {len(df)}")

    cities = df["city"]
    idx_to_drop = cities.str.contains(r"ñ|\/|\(|\)")
    cities = cities[~idx_to_drop]
    logger.info(f"Dropped {idx_to_drop.sum()} outliers")

    return list(cities)


def split_and_save(cities: List[str], out_dir: Path, random_state: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {out_dir}")

    with (out_dir / "full.txt").open("wt", encoding="utf-8") as f:
        f.write("\n".join(cities))

    train_set, val_set = train_test_split(
        cities, test_size=0.2, random_state=random_state
    )

    with (out_dir / "train.txt").open("wt", encoding="utf-8") as f:
        f.write("\n".join(train_set))
    with (out_dir / "val.txt").open("wt", encoding="utf-8") as f:
        f.write("\n".join(val_set))


@click.command()
@click.option("--input-path", "-i", required=True)
@click.option("--output-dir", "-o", required=True)
@click.option("--encoding", default="ISO-8859-1")
@click.option("--random-state", type=int, default=42)
def main(input_path: str, output_dir: str, encoding: str, random_state=42):
    logging.basicConfig(level=logging.INFO)

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    logger.info(f"Reading file {input_path} with encoding {encoding}")

    df = pd.read_csv(input_path, encoding=encoding)
    logger.info(f"Read {len(df)} rows")

    cities = preprocess_dataset(df)

    split_and_save(cities, output_dir, random_state)

    logger.info("Done.")


if __name__ == "__main__":
    main()
