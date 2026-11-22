import csv
import hashlib
import io

from library.lora_run_schema import HEADERS
from tools.lora_run_reader import read_csv


def test_hash_stops_at_end_of_complete_multiline_record(tmp_path):
    kind = "group_loss"
    sink = io.StringIO(newline="")
    writer = csv.writer(sink)
    writer.writerow(HEADERS[kind])
    writer.writerow([1, 2, "a\r\nb", 0, "", "", 2, "", ""])
    complete = sink.getvalue().encode("utf-8-sig")
    path = tmp_path / "group_loss.csv"
    path.write_bytes(complete + b'2,3,"unfinished\r\n')
    data = read_csv(path, kind)
    assert data.status == "partial" and len(data.rows) == 1
    assert data.source_bytes == len(complete) and data.sha256 == hashlib.sha256(complete).hexdigest()


def test_closed_header_only_is_an_observed_empty_csv(tmp_path):
    path = tmp_path / "gradient.csv"
    content = (",".join(HEADERS["gradient"]) + "\r\n").encode("utf-8-sig")
    path.write_bytes(content)
    data = read_csv(path, "gradient")
    assert data.status == "complete" and data.rows == []
    assert data.source_bytes == len(content) and data.sha256 == hashlib.sha256(content).hexdigest()
