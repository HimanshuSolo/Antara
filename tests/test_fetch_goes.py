from datetime import datetime
from pathlib import Path

from src.data import fetch_goes


class FakeS3Client:
    """Duck-typed stand-in for a boto3 S3 client: serves keys out of an
    in-memory list instead of hitting the real noaa-goes16 bucket."""

    def __init__(self, keys: list[str]):
        self.keys = keys
        self.downloaded: list[str] = []

    def list_objects_v2(self, Bucket, Prefix):
        matches = [k for k in self.keys if k.startswith(Prefix)]
        return {"Contents": [{"Key": k} for k in matches]}

    def download_file(self, Bucket, Key, Filename):
        self.downloaded.append(Key)
        Path(Filename).write_bytes(b"fake-scan-bytes")


def test_list_scans_filters_by_prefix_and_sorts():
    client = FakeS3Client([
        "ABI-L1b-RadF/2024/100/12/OR_ABI-L1b-RadF-M6C13_G16_s20241001210206_e1_c1.nc",
        "ABI-L1b-RadF/2024/100/12/OR_ABI-L1b-RadF-M6C13_G16_s20241001200206_e1_c1.nc",
        "ABI-L1b-RadF/2024/100/13/OR_ABI-L1b-RadF-M6C13_G16_s20241001300206_e1_c1.nc",  # wrong hour
        "ABI-L1b-RadF/2024/100/12/OR_ABI-L1b-RadF-M6C14_G16_s20241001210206_e1_c1.nc",  # wrong band
    ])

    keys = fetch_goes.list_scans(datetime(2024, 4, 9, 12), band=13, client=client)

    assert keys == [
        "ABI-L1b-RadF/2024/100/12/OR_ABI-L1b-RadF-M6C13_G16_s20241001200206_e1_c1.nc",
        "ABI-L1b-RadF/2024/100/12/OR_ABI-L1b-RadF-M6C13_G16_s20241001210206_e1_c1.nc",
    ]


def test_list_scans_range_spans_every_hour(monkeypatch):
    client = FakeS3Client([
        "ABI-L1b-RadF/2024/100/12/OR_ABI-L1b-RadF-M6C13_G16_s20241001200206_e1_c1.nc",
        "ABI-L1b-RadF/2024/100/13/OR_ABI-L1b-RadF-M6C13_G16_s20241001300206_e1_c1.nc",
    ])
    monkeypatch.setattr(fetch_goes, "_client", lambda: client)

    keys = fetch_goes.list_scans_range(
        datetime(2024, 4, 9, 12), datetime(2024, 4, 9, 13), band=13
    )

    assert len(keys) == 2


def test_download_skips_when_dest_already_exists(tmp_path):
    dest = tmp_path / "scan.nc"
    dest.write_bytes(b"already here")
    client = FakeS3Client([])

    result = fetch_goes.download("some/key/scan.nc", tmp_path, client=client)

    assert result == dest
    assert dest.read_bytes() == b"already here"
    assert client.downloaded == []


def test_download_fetches_missing_file(tmp_path):
    client = FakeS3Client([])

    result = fetch_goes.download("some/key/scan.nc", tmp_path, client=client)

    assert result == tmp_path / "scan.nc"
    assert result.read_bytes() == b"fake-scan-bytes"
    assert client.downloaded == ["some/key/scan.nc"]


def test_download_range_downloads_every_listed_scan(monkeypatch, tmp_path):
    client = FakeS3Client([
        "ABI-L1b-RadF/2024/100/12/OR_ABI-L1b-RadF-M6C13_G16_s20241001200206_e1_c1.nc",
        "ABI-L1b-RadF/2024/100/12/OR_ABI-L1b-RadF-M6C13_G16_s20241001210206_e1_c1.nc",
    ])
    monkeypatch.setattr(fetch_goes, "_client", lambda: client)

    paths = fetch_goes.download_range(
        datetime(2024, 4, 9, 12), datetime(2024, 4, 9, 12), band=13, dest_dir=tmp_path
    )

    assert len(paths) == 2
    assert all(p.exists() for p in paths)
    assert len(client.downloaded) == 2
