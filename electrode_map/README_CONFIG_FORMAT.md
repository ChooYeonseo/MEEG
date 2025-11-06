# Mosaic Configuration File Format

## Overview
Mosaic configuration files allow you to save and reuse electrode positions, channel mappings, and mosaic relationships for EEG analysis. This eliminates the need to repeatedly set up the same configuration.

## File Format
Configuration files are in JSON format with the following structure:

```json
{
  "electrode_positions": [
    {
      "number": 1,
      "x": 1.4,
      "y": -0.7
    },
    {
      "number": 2,
      "x": 0.8,
      "y": 0.2
    }
  ],
  "channel_mapping": {
    "1": "A-000",
    "2": "A-001"
  },
  "mosaic_relationships": [
    {
      "name": "mosaic 1",
      "electrode_a": 1,
      "electrode_b": 2
    }
  ],
  "metadata": {
    "description": "MEEG Mosaic Configuration",
    "version": "1.0",
    "total_electrodes": 2,
    "mapped_electrodes": 2,
    "relationships": 1
  }
}
```

## Field Descriptions

### electrode_positions (required)
An array of electrode objects, each containing:
- `number` (integer): Unique electrode identifier
- `x` (float): X-coordinate (Medial-Lateral position in mm)
- `y` (float): Y-coordinate (Anterior-Posterior position in mm)

### channel_mapping (optional)
A dictionary mapping electrode numbers to data channel names:
- Key: Electrode number (as string)
- Value: Channel name from your data file (e.g., "A-000", "A-001")

### mosaic_relationships (optional)
An array of relationship objects for mosaic analysis, each containing:
- `name` (string): User-friendly name for the relationship (e.g., "mosaic 1", "frontal pair")
- `electrode_a` (integer): First electrode number
- `electrode_b` (integer): Second electrode number

### metadata (optional)
Descriptive information about the configuration:
- `description`: Brief description
- `version`: Configuration format version
- `total_electrodes`: Number of electrodes
- `mapped_electrodes`: Number of mapped electrodes
- `relationships`: Number of mosaic relationships

## Usage

### Exporting Configuration
1. Open the MEEG application
2. Set up your electrode positions and mappings
3. Define mosaic relationships
4. Click "📤 Export Configuration"
5. Choose a location and filename (e.g., `my_mosaic_config.json`)

### Importing Configuration
1. Open the MEEG application
2. Click "📥 Import Configuration"
3. Select your saved configuration file
4. Confirm the import
5. All electrode positions, mappings, and relationships will be loaded

## Benefits
- **Reusability**: Save configurations for different experimental setups
- **Consistency**: Ensure the same configuration across multiple analyses
- **Sharing**: Share configurations with collaborators
- **Time-saving**: No need to manually recreate complex electrode arrangements

## Example Configurations
See the following example files in this directory:
- `Version2_Fixed.json`: Standard 19-electrode configuration
- `channel_mapping.json`: Custom channel mapping example

## Notes
- The configuration file format is compatible with both old tuple-based and new dictionary-based mosaic relationships
- When importing, the application validates the file structure and warns about any issues
- Electrode numbers must be unique within a configuration
- Channel names must match those available in your data files
