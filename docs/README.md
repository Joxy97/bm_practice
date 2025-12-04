# Documentation

This directory contains detailed documentation for the Boltzmann Machine Training Pipeline.

## Available Documentation

### [QUICKSTART.md](QUICKSTART.md)
**Quick Start Guide** - Get started quickly with step-by-step instructions:
- Installation and setup
- Running your first training
- Understanding the pipeline modes
- Basic configuration
- Common use cases

### [RUN_DIRECTORY_SYSTEM.md](RUN_DIRECTORY_SYSTEM.md)
**Run Directory System** - Experiment management and reproducibility:
- How the timestamped run directories work
- Viewing and managing past experiments
- Reproducing experiments
- Run directory structure
- Best practices for organization

## Adding New Documentation

When adding new documentation:

1. **Create the file in `docs/`**
   ```bash
   touch docs/NEW_FEATURE.md
   ```

2. **Add to this README**
   Update the "Available Documentation" section above

3. **Reference from main README**
   Add link in main `README.md` under the "Documentation" section

4. **Use relative paths**
   - From main README: `[docs/FILE.md](docs/FILE.md)`
   - Between docs: `[FILE.md](FILE.md)`
   - To config: `[../configs/config.yaml](../configs/config.yaml)`

## Documentation Standards

- Use clear, descriptive titles
- Include code examples
- Add table of contents for long documents
- Use relative links for cross-references
- Keep formatting consistent with existing docs
