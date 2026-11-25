"""Download/verify the official CC BY 4.0 FAR-Trans archive and extract four CSVs."""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

URL = "https://researchdata.gla.ac.uk/1658/1/FAR-Trans.zip"
ARCHIVE_SHA256 = "d0d6cfe027732bab45fdcf9af80eff4fefddc789aedae9bdce0d3d6cfd05cfb6"
CSV_SHA256 = {
    "asset_information.csv": "33f9897b14087068fba53a0bc9e24a0031b096bd3ad3e977e534153f29b7a590",
    "close_prices.csv": "6ba1fe8390b31dfb3ab4e3e824910f9e8da4684cb6074cdb82eb6422a323656d",
    "customer_information.csv": "c4ad52ebca90e32bf3df5eb64f362b7a3437b77d978b22627cc46927fe7064af",
    "transactions.csv": "358871fa1311a802799df86e56c3d633513566965b01412d058e0f8fbe6cd54b",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "FAR-Trans")
    parser.add_argument("--archive", type=Path, help="Use a downloaded official archive instead of the network")
    args = parser.parse_args()
    if all((args.data_dir / name).exists() and hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest() == digest
           for name, digest in CSV_SHA256.items()):
        print("Four existing CSVs match the pinned official archive.")
        return
    archive = args.archive or args.data_dir.parent / "FAR-Trans.zip"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(URL, headers={"User-Agent": "FAR-Trans-HW2-reproduction"})
        with urllib.request.urlopen(request, timeout=180) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != ARCHIVE_SHA256:
            raise ValueError("Official archive changed; inspect provenance before using a different dataset")
        archive.write_bytes(content)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Archive hash does not match the experiment")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for name, expected in CSV_SHA256.items():
            matches = [member for member in source.namelist() if Path(member).name == name]
            if len(matches) != 1:
                raise ValueError("Expected one archive entry for " + name)
            data = source.read(matches[0])
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("CSV hash mismatch for " + name)
            (args.data_dir / name).write_bytes(data)
    print("Extracted four verified CSVs to", args.data_dir)


if __name__ == "__main__":
    main()
