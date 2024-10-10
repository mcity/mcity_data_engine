r"""
# Mcity Data Engine

The Mcity Data Engine is an essential tool in the Mcity makerspace for transportation innovators making AI algorithms and seeking actionable data insights through machine learning.

Learn more: https://mcity.umich.edu/what-we-do/data-for-ai/
# Overview

Our Data Engine allows researchers to efficiently query large volumes of data – directing Mcity’s transportation data collection networks with specific tasks to produce high-quality artificial intelligence models for development.

Using the power of AI, the Mcity Data Engine provides our member companies and researchers with fine-tuned data sets for mobility applications through a continuously-improving loop that uses data and labels to optimize and improve the performance of AI algorithms, delivering only the specific data wanted.

The Mcity Data Engine is the first tool of its kind to be made available to Mcity Members, including early stage companies. The Mcity Data Engine:

- Provides access to a large compute cluster tailored to AI training for mobility applications
- Efficiently collects, processes and labels vast networks of Mcity data
- Interfaces with Mcity data and models to build AI-powered solutions
- Develops large, diverse, annotated training data sets for vehicle testing
- Boosts training data sets with computer simulations to create and add synthetic data
- Runs at computer speeds when automated, operating 24/7
- Serves as building block for other mobility transportation projects
"""

from __future__ import annotations

__docformat__ = "markdown"  # explicitly disable rST processing in the examples above.

from pathlib import Path
from typing import overload

from pdoc import doc
from pdoc import extract
from pdoc import render


@overload
def pdoc(
    *modules: Path | str,
    output_directory: None = None,
) -> str:
    pass


@overload
def pdoc(
    *modules: Path | str,
    output_directory: Path,
) -> None:
    pass


def pdoc(
    *modules: Path | str,
    output_directory: Path | None = None,
) -> str | None:
    """
    Render the documentation for a list of modules.

     - If `output_directory` is `None`, returns the rendered documentation
       for the first module in the list.
     - If `output_directory` is set, recursively writes the rendered output
       for all specified modules and their submodules to the target destination.

    Rendering options can be configured by calling `pdoc.render.configure` in advance.
    """
    all_modules: dict[str, doc.Module] = {}
    for module_name in extract.walk_specs(modules):
        all_modules[module_name] = doc.Module.from_name(module_name)

    for module in all_modules.values():
        out = render.html_module(module, all_modules)
        if not output_directory:
            return out
        else:
            outfile = output_directory / f"{module.fullname.replace('.', '/')}.html"
            outfile.parent.mkdir(parents=True, exist_ok=True)
            outfile.write_bytes(out.encode())

    assert output_directory

    index = render.html_index(all_modules)
    if index:
        (output_directory / "index.html").write_bytes(index.encode())

    search = render.search_index(all_modules)
    if search:
        (output_directory / "search.js").write_bytes(search.encode())

    return None
