#!/usr/bin/env python3
"""
Semantic File System Terminal - Instant Search (No LLM in loop)
"""

import os
import sys
import textwrap
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich import box
from rich.markdown import Markdown
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from lsfs.config import LSFSConfig
from lsfs.file_system import LocalLSFS

console = Console()

def display_banner():
    """Display welcome banner"""
    banner = """
    # 🚀 Semantic File System v2.0
    ### Instant Search • No LLM Overhead • CSV Indexing
    """
    console.print(Panel(Markdown(banner), border_style="cyan", box=box.DOUBLE))

def display_help():
    """Display available commands"""
    help_table = Table(title="Available Commands", box=box.ROUNDED)
    help_table.add_column("Command", style="cyan", no_wrap=True)
    help_table.add_column("Description", style="white")
    help_table.add_column("Example", style="yellow")
    
    commands = [
        ("search <query>", "Search files by content (instant)", "search python functions"),
        ("find <query>", "Alias for search", "find machine learning"),
        ("list [dir]", "List files in directory", "list documents"),
        ("read <file>", "Read file content", "read notes.txt"),
        ("create file <name>", "Create new file", "create file test.txt"),
        ("create dir <name>", "Create directory", "create dir projects"),
        ("delete <name>", "Delete file/directory", "delete old.txt"),
        ("index", "Re-index all files", "index"),
        ("status", "Show system status", "status"),
        ("help", "Show this help", "help"),
        ("exit/quit", "Exit the application", "exit")
    ]
    
    for cmd, desc, example in commands:
        help_table.add_row(cmd, desc, example)
    
    console.print(help_table)

def display_search_results(results: list):
    """Display search results in a table"""
    if not results:
        console.print("[yellow]No results found[/yellow]")
        return
    
    table = Table(title=f"Found {len(results)} file(s)", box=box.ROUNDED)
    table.add_column("#", style="cyan", width=4)
    table.add_column("File Name", style="green")
    table.add_column("Path", style="white")
    table.add_column("Similarity", style="magenta", justify="right")
    table.add_column("Size", style="yellow", justify="right")
    table.add_column("Preview", style="white")
    
    for i, result in enumerate(results, 1):
        similarity_pct = f"{result['similarity'] * 100:.1f}%"
        size_kb = f"{result['size'] / 1024:.1f} KB"
        preview = result.get("preview", "")
        preview = textwrap.shorten(preview, width=80, placeholder="...")
        
        table.add_row(
            str(i),
            result['file_name'],
            result['relative_path'],
            similarity_pct,
            size_kb,
            preview
        )
    
    console.print(table)

def display_file_list(data: dict):
    """Display file listing"""
    table = Table(title=f"Contents of: {data['path']}", box=box.ROUNDED)
    table.add_column("Type", style="cyan", width=8)
    table.add_column("Name", style="white")
    table.add_column("Size", style="yellow", justify="right")
    table.add_column("Modified", style="magenta")
    
    # Directories first
    for dir_name in data.get('directories', []):
        table.add_row("📁 DIR", dir_name, "-", "-")
    
    # Then files
    for file in data.get('files', []):
        size_kb = f"{file['size'] / 1024:.1f} KB"
        modified = file['modified'].split('T')[0]  # Date only
        table.add_row("📄 FILE", file['name'], size_kb, modified)
    
    console.print(table)
    console.print(f"[cyan]Total: {data['total']} items[/cyan]")

def display_stats(data: dict):
    """Display system statistics"""
    stats_table = Table(title="System Statistics", box=box.ROUNDED)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    
    vector_stats = data.get('vector_stats', {})
    
    stats_table.add_row("Root Directory", data['root_dir'])
    stats_table.add_row("Total Indexed Files", str(vector_stats.get('total_files', 0)))
    stats_table.add_row("CSV Index Path", vector_stats.get('csv_index', 'N/A'))
    stats_table.add_row("Database Path", vector_stats.get('db_path', 'N/A'))
    collections = vector_stats.get('collections', {})
    if collections:
        for name, count in collections.items():
            stats_table.add_row(f"Collection: {name}", str(count))
    
    console.print(stats_table)


def _short_path(path: str, max_len: int = 60) -> str:
    if not path:
        return ""
    if len(path) <= max_len:
        return path
    return "..." + path[-(max_len - 3) :]


def run_reindex(lsfs: LocalLSFS):
    """Run reindex with live progress UI."""
    progress = Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=None, style="green"),
        TextColumn("[yellow]{task.completed}/{task.total}"),
        TextColumn("[magenta]{task.fields[stats]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    stats_snapshot = {"indexed": 0, "unchanged": 0, "skipped": 0, "error": 0}

    def callback(current, total, stats, file_path, _result):
        stats_snapshot.update(stats)
        stats_text = (
            f"idx:{stats_snapshot['indexed']} "
            f"un:{stats_snapshot['unchanged']} "
            f"sk:{stats_snapshot['skipped']} "
            f"err:{stats_snapshot['error']}"
        )
        desc = (
            f"Indexing {_short_path(file_path, 60) if file_path else ''}"
        )
        progress.update(
            task_id,
            total=total,
            completed=current,
            description=desc,
            stats=stats_text,
        )

    with progress:
        task_id = progress.add_task("Indexing", total=0, stats="idx:0 un:0 sk:0 err:0")
        result = lsfs.reindex_all(progress=callback)

    return result

def main():
    """Main terminal loop"""
    try:
        # Load configuration
        config = LSFSConfig.from_yaml("config.yaml")
        
        # Initialize file system
        console.print("[cyan]Initializing Semantic File System...[/cyan]")
        lsfs = LocalLSFS(config)
        
        # Display banner
        display_banner()
        console.print("[green]✓ System ready! Type 'help' for commands[/green]\n")
        
        # Main loop
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("[bold cyan]LSFS[/bold cyan]").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    console.print("[yellow]Goodbye! 👋[/yellow]")
                    break
                
                if user_input.lower() == 'help':
                    display_help()
                    continue
                
                if user_input.lower() == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    display_banner()
                    continue
                
                # Execute command (NO LLM OVERHEAD)
                if user_input.lower() in ["index", "reindex"]:
                    result = run_reindex(lsfs)
                else:
                    with console.status("[cyan]Processing...[/cyan]", spinner="dots"):
                        result = lsfs.parse_and_execute(user_input)
                
                # Display results
                if result.get('success'):
                    # Search results
                    if 'results' in result:
                        display_search_results(result['results'])
                    
                    # File listing
                    elif 'files' in result:
                        display_file_list(result)
                    
                    # Stats
                    elif 'vector_stats' in result:
                        display_stats(result)
                    
                    # File content
                    elif 'content' in result:
                        console.print(Panel(
                            result['content'][:500] + ("..." if len(result['content']) > 500 else ""),
                            title=f"[green]{result['file_name']}[/green]",
                            border_style="green"
                        ))
                    
                    # Generic success message
                    elif 'message' in result:
                        console.print(f"[green]✓ {result['message']}[/green]")
                    
                else:
                    console.print(f"[red]✗ Error: {result.get('error', 'Unknown error')}[/red]")
            
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")
    
    except Exception as e:
        console.print(f"[red]Fatal error: {str(e)}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
