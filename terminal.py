#!/usr/bin/env python3
"""
Semantic File System Terminal - Instant Search (No LLM in loop)
"""

import os
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich import box
from rich.markdown import Markdown
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
    
    for i, result in enumerate(results, 1):
        similarity_pct = f"{result['similarity'] * 100:.1f}%"
        size_kb = f"{result['size'] / 1024:.1f} KB"
        
        table.add_row(
            str(i),
            result['file_name'],
            result['relative_path'],
            similarity_pct,
            size_kb
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
    
    console.print(stats_table)

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
