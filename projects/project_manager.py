"""
Project Manager - CLI tool for creating and managing BM projects.

Usage:
    python -m projects.project_manager create --name my_project
    python -m projects.project_manager list
"""

import os
import shutil
import argparse
from pathlib import Path


class ProjectManager:
    """Manages BM project lifecycle."""

    def __init__(self, projects_dir: str = "projects"):
        self.projects_dir = Path(projects_dir)

    def create_project(self, project_name: str, template: str = "template"):
        """
        Create new project from template.

        Args:
            project_name: Name for the new project
            template: Template to use (default: "template")
        """
        project_path = self.projects_dir / project_name
        template_path = self.projects_dir / template

        if project_path.exists():
            print(f"Error: Project '{project_name}' already exists at {project_path}")
            return

        if not template_path.exists():
            print(f"Error: Template '{template}' not found at {template_path}")
            return

        # Copy template
        print(f"Creating project '{project_name}' from template '{template}'...")
        shutil.copytree(template_path, project_path)

        print(f"\n✓ Project '{project_name}' created successfully!")
        print(f"\nProject location: {project_path}")
        print(f"\nNext steps:")
        print(f"  1. Edit configuration:")
        print(f"     {project_path}/project_config.py")
        print(f"  2. Implement custom dataset:")
        print(f"     {project_path}/custom_dataset.py")
        print(f"  3. Place your CSV data files in:")
        print(f"     {project_path}/data/")
        print(f"  4. Run training:")
        print(f"     python -m bm_core.bm --mode train --config {project_path}/project_config.py --dataset {project_path}/data/train.csv")

    def list_projects(self):
        """List all projects."""
        projects = [
            d.name for d in self.projects_dir.iterdir()
            if d.is_dir() and d.name not in ['template', '__pycache__']
        ]

        if not projects:
            print("No projects found.")
            print(f"\nCreate a project with:")
            print(f"  python -m projects.project_manager create --name my_project")
        else:
            print(f"\nAvailable projects ({len(projects)}):")
            for p in sorted(projects):
                print(f"  - {p}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="BM Project Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'action',
        choices=['create', 'list'],
        help='Action to perform'
    )

    parser.add_argument(
        '--name',
        type=str,
        help='Project name (for create action)'
    )

    parser.add_argument(
        '--template',
        type=str,
        default='template',
        help='Template to use (default: template)'
    )

    args = parser.parse_args()

    manager = ProjectManager()

    if args.action == 'create':
        if not args.name:
            print("Error: --name is required for create action")
            return
        manager.create_project(args.name, args.template)

    elif args.action == 'list':
        manager.list_projects()


if __name__ == '__main__':
    main()
