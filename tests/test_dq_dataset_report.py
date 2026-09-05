import base64
import copy
import io
import json

from PIL import Image

from dq_profile.diagnostic_report import report_thumbnails, write_report


def test_thumbnail_preserves_aspect_alpha_and_survives_source_removal(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGBA", (1200, 600), (255, 0, 0, 0)).save(source)
    rows = [dict(image_id="one", path=str(source), measured=True),
            dict(image_id="one", path=str(source), measured=True),
            dict(image_id="unmeasured", path=str(source), measured=False)]
    previews = report_thumbnails(tmp_path, rows)
    assert set(previews) == {"one"}
    value = previews["one"]
    with Image.open(io.BytesIO(base64.b64decode(value["data_url"].split(",", 1)[1]))) as img:
        assert img.size == (512, 256)
        assert min(img.getpixel((0, 0))) >= 250
        assert not img.getexif()
    source.unlink()
    assert report_thumbnails(tmp_path, rows) == previews


def test_thumbnail_uses_exif_orientation(tmp_path):
    source = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (1200, 600), "blue").save(source, exif=exif)
    preview = report_thumbnails(tmp_path, [dict(image_id="rotated", path=str(source), measured=True)])["rotated"]
    assert (preview["width"], preview["height"]) == (256, 512)


def test_missing_or_corrupt_images_do_not_stop_report_and_can_be_retried(tmp_path):
    missing = tmp_path / "missing.png"
    corrupt = tmp_path / "bad.png"
    corrupt.write_bytes(b"invalid image")
    rows = [dict(image_id="missing", path=str(missing), measured=True),
            dict(image_id="corrupt", path=str(corrupt), measured=True)]
    values = report_thumbnails(tmp_path, rows)
    assert values["missing"]["status"] == "missing"
    assert values["corrupt"]["status"] == "unreadable"
    Image.new("RGB", (20, 10)).save(missing)
    assert report_thumbnails(tmp_path, rows)["missing"]["status"] == "available"


def test_report_embeds_preview_without_mutating_numerical_payload(tmp_path):
    source = tmp_path / "base.png"
    Image.new("RGB", (12, 8), "blue").save(source)
    payload = {"samples": [dict(image_id="image", path=str(source), measured=True)],
               "all": {"loss_pre": .2, "loss_post": .1}, "caption": "</script>"}
    original = copy.deepcopy(payload)
    report = tmp_path / "report.html"
    write_report(report, payload)
    assert payload == original
    text = report.read_text(encoding="utf-8")
    data = json.loads(text.split('<script id="payload" type="application/json">')[1].split('</script>')[0])
    assert data["thumbnails"]["image"]["data_url"].startswith("data:image/jpeg;base64,")
    assert data["all"] == payload["all"]
    assert "</script>" not in text.split('<script id="payload" type="application/json">')[1].split('</script>')[0]
