# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the FAIR Noncommercial Research License
# found in the LICENSE file in the root directory of this source tree.

import json

from .stsp_constants import STSP_DATA_ROOT, STSP_MANIFEST_ROOT


def check_all_datasets():
    for dataset_manifset in STSP_MANIFEST_ROOT.glob("**/*jsonl"):
        records_checked = 0
        print(f"dataset_manifset: {dataset_manifset}")
        with dataset_manifset.open() as f:
            for record in f:
                record = json.loads(record)
                for wav_key in ["wav_path", "prompt", "generation"]:
                    if wav_key in record and record[wav_key].endswith(".wav"):
                        wav_path = STSP_DATA_ROOT / record[wav_key]
                    assert (
                        wav_path.is_file()
                    ), f"Record {record[wav_key]} not found in {str(wav_path)} and listed in {dataset_manifset}"
                records_checked += 1
        print(f"{records_checked} records checked for {dataset_manifset.stem} split")


if __name__ == "__main__":
    check_all_datasets()
