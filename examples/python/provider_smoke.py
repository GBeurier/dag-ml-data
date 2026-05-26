from __future__ import annotations

import argparse
import json

from dag_ml_data_provider import InMemoryProvider


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", required=True)
    parser.add_argument("--envelope", required=True)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()

    target_tables = [
        {
            "target_id": "y",
            "values": [
                {"sample_id": "S001", "value": 42.0},
                {"sample_id": "S002", "value": 7.0},
            ],
        }
    ]
    with InMemoryProvider.from_files(args.lib, args.envelope, target_tables) as provider:
        data_handle = provider.materialize_file(args.request)
        view_handle = provider.make_view(
            data_handle,
            {"sample_ids": ["S001"], "include_augmented": False},
        )
        identity = provider.view_identity(view_handle)
        targets = provider.target_values(view_handle, "y")

        assert [row["observation_id"] for row in identity] == [
            "obs.S001.base",
            "obs.S001.rep1",
        ]
        assert targets == [{"sample_id": "S001", "target_id": "y", "value": 42.0}]

        provider.release(view_handle)
        provider.release(data_handle)

    print(
        json.dumps(
            {
                "identity_rows": len(identity),
                "target_rows": len(targets),
                "observations": [row["observation_id"] for row in identity],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
