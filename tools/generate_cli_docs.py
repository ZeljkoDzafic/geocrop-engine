"""Regenerate docs/cli.md from the argparse definitions:  python tools/generate_cli_docs.py"""
import argparse

from geocrop.cli import build_parser

p = build_parser()
sub = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
help_by = {a.dest: a.help for a in sub._choices_actions}
out = ["# CLI reference", "", "Generated from `geocrop --help`. Every command prints JSON or a path; run "
       "`geocrop <command> --help` for details.", ""]
for name, sp in sub.choices.items():
    out += [f"## `geocrop {name}`", "", help_by.get(name, "") or "", ""]
    rows = []
    for a in sp._actions:
        if isinstance(a, argparse._HelpAction):
            continue
        flag = ", ".join(a.option_strings) or a.dest
        req = "yes" if getattr(a, "required", False) else ""
        default = "" if a.default in (None, argparse.SUPPRESS, False) else f"`{a.default}`"
        rows.append(f"| `{flag}` | {req} | {default} | {(a.help or '').replace('|', '/')} |")
    if rows:
        out += ["| Option | Required | Default | Notes |", "|---|---|---|---|"] + rows + [""]
    nested = next((a for a in sp._actions if isinstance(a, argparse._SubParsersAction)), None)
    for sub_name, ssp in (nested.choices.items() if nested else ()):
        out += [f"### `geocrop {name} {sub_name}`", "", ssp.description or "", ""]
        srows = []
        for a in ssp._actions:
            if isinstance(a, argparse._HelpAction):
                continue
            flag = ", ".join(a.option_strings) or a.dest
            req = "yes" if getattr(a, "required", False) else ""
            default = "" if a.default in (None, argparse.SUPPRESS, False) else f"`{a.default}`"
            srows.append(f"| `{flag}` | {req} | {default} | {(a.help or '').replace('|', '/')} |")
        if srows:
            out += ["| Option | Required | Default | Notes |", "|---|---|---|---|"] + srows + [""]
open("docs/cli.md", "w").write("\n".join(out))
