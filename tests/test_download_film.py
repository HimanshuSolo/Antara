from src.deep.download_film import download


def test_download_skips_when_dest_already_exists(tmp_path):
    dest = tmp_path / "film_net_fp32.pt"
    dest.write_bytes(b"already here")

    # No network access should be attempted -- if it were, this would
    # fail/hang against the bogus URL below.
    result = download("http://example.invalid/not-a-real-checkpoint", dest)

    assert result == dest
    assert dest.read_bytes() == b"already here"
