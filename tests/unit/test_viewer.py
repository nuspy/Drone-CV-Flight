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
