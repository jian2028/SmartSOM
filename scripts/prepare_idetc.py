"""Export the verified frozen IDETC fixture into existing SmartSOM configurations."""

import argparse
from pathlib import Path

from validation.idetc_inputs import write_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = write_bundle(args.output_dir)
    print(
        f"converted {len(result['cases'])} frozen cases; study={args.output_dir / 'study.yaml'}"
    )


if __name__ == "__main__":
    main()
