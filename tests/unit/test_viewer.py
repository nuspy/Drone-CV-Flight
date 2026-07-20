"""Realtime 3D viewer: HTML emit + local server for scene.glb."""

from __future__ import annotations

from dronecv.gis.export.viewer_html import VIEWER_HTML, write_viewer


def test_write_viewer_emits_html(tmp_path):
    p = write_viewer(tmp_path)
    assert p.name == "viewer.html"
    html = p.read_text()
    # loads the model + meta from its own dir (must be served, not file://)
    assert "scene.glb" in html and "scene_meta.json" in html
    # the exact controls the user asked for are wired
    for token in ("KeyW", "KeyS", "KeyA", "KeyD", "KeyQ", "KeyE",
                  "KeyG", "wheel", "Tab", "ShiftLeft"):
        assert token in html
    # sky + sun + shadows are set up
    assert "Sky" in html and "DirectionalLight" in html and "shadowMap" in html


def test_viewer_supports_open_and_screenshot(tmp_path):
    html = write_viewer(tmp_path).read_text()
    # open arbitrary files: query param, file input, Open button
    assert "?model=" not in html  # not hardcoded
    assert "get('model')" in html and 'type="file"' in html and "openbtn" in html
    # screenshot: POSTs to the server and encodes coords + height in the name
    assert "screenshot" in html and "toDataURL" in html
    assert "camGeo" in html and "lat" in html and "lon" in html and "_h" in html
    assert "preserveDrawingBuffer: true" in html


def test_viewer_html_is_selfcontained_module():
    # three is pulled via an importmap; the page is a single document.
    assert VIEWER_HTML.strip().startswith("<!doctype html>")
    assert 'type="importmap"' in VIEWER_HTML


def test_serve_scene_dir_serves_files(tmp_path):
    import urllib.request

    from dronecv.commands.gis_cmd import serve_scene_dir

    write_viewer(tmp_path)
    (tmp_path / "scene.glb").write_bytes(b"glTF-bytes")
    httpd, url = serve_scene_dir(tmp_path)
    try:
        assert url.endswith("/viewer.html")
        body = urllib.request.urlopen(url, timeout=5).read().decode()
        assert "DroneCV scene" in body
        glb_url = url.rsplit("/", 1)[0] + "/scene.glb"
        assert urllib.request.urlopen(glb_url, timeout=5).read() == b"glTF-bytes"
    finally:
        httpd.shutdown()


def test_serve_scene_dir_model_query(tmp_path):
    from dronecv.commands.gis_cmd import serve_scene_dir

    httpd, url = serve_scene_dir(tmp_path, model="my model.glb")
    try:
        assert url.endswith("/viewer.html?model=my%20model.glb")
    finally:
        httpd.shutdown()


def test_screenshot_post_saves_under_model_dir(tmp_path):
    import base64
    import json
    import urllib.request

    from dronecv.commands.gis_cmd import serve_scene_dir

    # a tiny valid PNG (1x1) as a data URL
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    data_url = "data:image/png;base64," + base64.b64encode(png).decode()
    httpd, url = serve_scene_dir(tmp_path)
    try:
        base = url.rsplit("/", 1)[0]
        payload = json.dumps({
            "model": "budapest_hq",
            "filename": "budapest_hq_lat47.500000_lon19.040000_h150.0m.png",
            "data": data_url,
        }).encode()
        req = urllib.request.Request(base + "/screenshot", data=payload,
                                     headers={"Content-Type": "application/json"})
        resp = json.loads(urllib.request.urlopen(req, timeout=5).read())
        saved = tmp_path / "screenshots" / "budapest_hq" / \
            "budapest_hq_lat47.500000_lon19.040000_h150.0m.png"
        assert saved.exists() and saved.read_bytes() == png
        assert resp["path"] == str(saved)
    finally:
        httpd.shutdown()


def test_screenshot_post_sanitizes_paths(tmp_path):
    import base64
    import json
    import urllib.request

    from dronecv.commands.gis_cmd import serve_scene_dir

    data_url = "data:image/png;base64," + base64.b64encode(b"x").decode()
    httpd, url = serve_scene_dir(tmp_path)
    try:
        base = url.rsplit("/", 1)[0]
        payload = json.dumps({
            "model": "../../etc", "filename": "../../evil", "data": data_url,
        }).encode()
        req = urllib.request.Request(base + "/screenshot", data=payload,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5).read()
        # nothing escaped the served tree
        assert not (tmp_path.parent / "evil").exists()
        files = list((tmp_path / "screenshots").rglob("*.png"))
        assert len(files) == 1
        assert tmp_path in files[0].parents
    finally:
        httpd.shutdown()
