from pathlib import Path
import mkdocs_gen_files as gen

pkg = "repo_name"
root = Path("src") / pkg

nav = gen.Nav()

for path in sorted(root.rglob("*.py")):
    if path.name == "__init__.py":
        continue
    mod = ".".join((pkg, *path.relative_to(root).with_suffix("").parts))
    doc_path = Path("api", *path.relative_to(root).with_suffix(".md").parts)
    nav[tuple(doc_path.with_suffix("").parts)] = doc_path.as_posix()
    with gen.open(doc_path, "w") as f:
        f.write(f"# `{mod}`\n\n::: {mod}\n")

with gen.open("api/SUMMARY.md", "w") as f:
    f.write(nav.build_literate_nav())