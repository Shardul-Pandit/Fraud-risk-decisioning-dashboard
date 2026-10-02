from streamlit.testing.v1 import AppTest

APP = "app/streamlit_app.py"


def test_app_opens_on_a_finished_demo_review(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    app = AppTest.from_file(APP, default_timeout=120).run()

    assert not app.exception
    assert app.radio[1].value == "Demo Transaction"
    assert any("classified as High Risk" in m.value for m in app.markdown)
    assert any("rule-based template" in c.value for c in app.caption)


def test_other_pages_render(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    app = AppTest.from_file(APP, default_timeout=120).run()

    for page in ["Model Performance", "About This App"]:
        app.radio[0].set_value(page).run()
        assert not app.exception
