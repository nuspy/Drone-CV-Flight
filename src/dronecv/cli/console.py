from __future__ import annotations

from rich.console import Console
from rich.table import Table

console = Console()


def metric_table(title: str, rows: dict[str, str]) -> Table:
    table = Table(title=title, show_header=False)
    table.add_column(style="bold cyan")
    table.add_column()
    for key, value in rows.items():
        table.add_row(key, value)
    return table
