"""Downloader tests: checksum verification, idempotent skip, extraction."""

import hashlib
import zipfile

import pytest

from provenance_lens.data.download import (
    ChecksumMismatchError,
    Source,
    SourceFile,
    ensure_file,
    load_sources,
    process_source,
    sha256_of,
)


@pytest.fixture()
def fixture_archive(tmp_path):
    payload_dir = tmp_path / "payload"
    payload_dir.mkdir()
    (payload_dir / "a.jpg").write_bytes(b"fake-image-a")
    (payload_dir / "b.jpg").write_bytes(b"fake-image-b")
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted(payload_dir.iterdir()):
            zf.write(path, arcname=path.name)
    return archive


def _source_for(archive, sha):
    return Source(
        name="fixture",
        kind="archive",
        license="test license",
        classes=["authentic"],
        enabled=True,
        files=[SourceFile(url=archive.as_uri(), sha256=sha)],
    )


def test_checksum_verified_and_extracted(tmp_path, fixture_archive):
    source = _source_for(fixture_archive, sha256_of(fixture_archive))
    record = process_source(source, tmp_path / "raw")
    assert record["extracted_files"] == 2
    assert (tmp_path / "raw" / "fixture" / "a.jpg").read_bytes() == b"fake-image-a"
    record_file = tmp_path / "raw" / "fixture" / "source_record.json"
    assert record_file.exists()
    assert "test license" in record_file.read_text()


def test_corrupt_archive_fails_with_both_hashes(tmp_path, fixture_archive):
    wrong = hashlib.sha256(b"not-the-file").hexdigest()
    source = _source_for(fixture_archive, wrong)
    with pytest.raises(ChecksumMismatchError) as excinfo:
        process_source(source, tmp_path / "raw")
    assert wrong in str(excinfo.value)
    assert sha256_of(fixture_archive) in str(excinfo.value)


def test_verified_file_is_not_refetched(tmp_path, fixture_archive):
    sha = sha256_of(fixture_archive)
    source_file = SourceFile(url=fixture_archive.as_uri(), sha256=sha)
    archive_dir = tmp_path / "archives"
    first = ensure_file(source_file, archive_dir)
    assert first.exists()
    fixture_archive.unlink()  # a second fetch would now fail loudly
    second = ensure_file(source_file, archive_dir)
    assert second == first


def test_project_config_parses():
    sources = load_sources(__import__("pathlib").Path("configs/data_sources.yaml"))
    names = {s.name for s in sources}
    assert {"micc_f220", "micc_f2000", "cifake", "casia_v2"} <= names
    for source in sources:
        assert source.license
        if source.enabled:
            assert source.files, f"enabled source {source.name} has no files"


def test_truncated_download_raises(tmp_path, monkeypatch):
    import email.message
    import io as _io

    from provenance_lens.data import download as dl

    headers = email.message.Message()
    headers["Content-Length"] = "100"

    class FakeResponse(_io.BytesIO):
        def __init__(self):
            super().__init__(b"only-30-bytes-of-a-promised-100")
            self.headers = headers

        def __exit__(self, *args):
            self.close()

        def __enter__(self):
            return self

    monkeypatch.setattr(dl.urllib.request, "urlopen", lambda url: FakeResponse())
    with pytest.raises(dl.TruncatedDownloadError) as excinfo:
        dl.fetch("http://example.invalid/file.zip", tmp_path / "file.zip")
    assert "100" in str(excinfo.value)
    assert not (tmp_path / "file.zip").exists()


def test_second_run_skips_extraction(tmp_path, fixture_archive, capsys):
    source = _source_for(fixture_archive, sha256_of(fixture_archive))
    first = process_source(source, tmp_path / "raw")
    second = process_source(source, tmp_path / "raw")
    assert second == first
    assert "already extracted" in capsys.readouterr().out
