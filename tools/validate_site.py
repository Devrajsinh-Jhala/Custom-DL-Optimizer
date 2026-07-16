from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


class _DocumentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.references: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name: value for name, value in attrs if value is not None}
        if "id" in values:
            self.ids.append(values["id"])
        if tag in {"a", "link"} and "href" in values:
            self.references.append((tag, values["href"]))
        if tag in {"img", "script", "source"} and "src" in values:
            self.references.append((tag, values["src"]))


def _parse(path: Path) -> _DocumentParser:
    parser = _DocumentParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


def main() -> int:
    documents = {path.resolve(): _parse(path) for path in SITE.glob("*.html")}
    errors: list[str] = []
    for path, document in documents.items():
        duplicates = sorted({value for value in document.ids if document.ids.count(value) > 1})
        if duplicates:
            errors.append(f"{path.name}: duplicate ids: {', '.join(duplicates)}")
        for tag, reference in document.references:
            split = urlsplit(reference)
            if split.scheme or split.netloc or reference.startswith(("mailto:", "tel:")):
                continue
            relative_path = unquote(split.path)
            target = (path.parent / relative_path).resolve() if relative_path else path
            if relative_path.endswith("/"):
                target = target / "index.html"
            if not target.exists():
                errors.append(f"{path.name}: missing {tag} target {reference!r}")
                continue
            if split.fragment and target.suffix.lower() == ".html":
                target_document = documents.get(target) or _parse(target)
                if split.fragment not in target_document.ids:
                    errors.append(
                        f"{path.name}: missing fragment {split.fragment!r} in {target.name}"
                    )

    index_text = (SITE / "index.html").read_text(encoding="utf-8")
    guides_text = (SITE / "guides.html").read_text(encoding="utf-8")
    for name, text in (("index.html", index_text), ("guides.html", guides_text)):
        if "v3" not in text and "Version 3" not in text:
            errors.append(f"{name}: v3 release copy is missing")

    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Validated {len(documents)} HTML documents and their local references.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
