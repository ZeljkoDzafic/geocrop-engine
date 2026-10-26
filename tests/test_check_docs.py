"""tools/check_docs.py keeps the documentation consistent with the repository, the CLI and the config defaults."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_docs", ROOT / "tools/check_docs.py")
check_docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_docs)


def test_repository_docs_are_consistent(capsys):
    assert check_docs.main([]) == 0, capsys.readouterr().out


def test_cli_reference_is_generated_from_parser(tmp_path, monkeypatch):
    import runpy

    monkeypatch.chdir(tmp_path)
    (tmp_path / "docs").mkdir()
    runpy.run_path(str(ROOT / "tools/generate_cli_docs.py"))
    generated = (tmp_path / "docs/cli.md").read_text(encoding="utf8")
    assert generated == (ROOT / "docs/cli.md").read_text(encoding="utf8"), \
        "docs/cli.md is stale: run python tools/generate_cli_docs.py"


def test_detects_broken_links_commands_and_config_keys(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text(
        "# Title\n\n```bash\ngeocrop train --samples s.npz --no-such-option 1 \\\n  --out m\n"
        "geocrop cube no-such-sub\n```\n\n[a](missing.md) [b](#title) [c](#no-anchor) `geocrop no-such-cmd`\n"
        "`from geocrop import config`\n", encoding="utf8")
    (tmp_path / "docs/configuration.md").write_text(
        "| Section | Key | Default | Meaning |\n|---|---|---|---|\n| `model` | `no_such_key`, `cv_folds` | 1 | x |\n",
        encoding="utf8")
    links = check_docs.check_links(tmp_path)
    assert any("missing.md" in e for e in links) and any("#no-anchor" in e for e in links)
    assert not any("#title" in e for e in links)
    commands = "\n".join(check_docs.check_commands(tmp_path))
    assert "--no-such-option" in commands and "cube no-such-sub" in commands and "no-such-cmd" in commands
    assert "import" not in commands
    config_errors = "\n".join(check_docs.check_config_keys(tmp_path))
    assert "'model.no_such_key' is not in" in config_errors
    assert "'model.n_estimators' (geocrop.config.DEFAULTS) is not documented" in config_errors
    assert "model.cv_folds'" not in config_errors
