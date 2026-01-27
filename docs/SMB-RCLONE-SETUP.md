# SMB Backup Setup with Rclone

This guide helps configure rclone to reliably copy files to your SMB share (`ramanfile.local`).

## Quick Start

### 1. Install Rclone (Already Done)
```bash
rclone version
```

### 2. Configure SMB Remote

Run the interactive configuration:
```bash
rclone config
```

When prompted:
- **Select option**: `n` (New remote)
- **Name**: `SMB-Ramanfile`
- **Storage type**: Search for `smb` and select it
- **SMB host**: `ramanfile.local`
- **SMB username**: `ramanlab`
- **SMB password**: Enter your password (will be encrypted)
- **SMB port**: `445` (press Enter for default)
- **Domain**: Leave empty (press Enter)
- **Confirm settings**: `y`

### 3. Test Connection
```bash
rclone listremotes        # Should show SMB-Ramanfile
rclone lsd SMB-Ramanfile: # Test listing SMB shares
```

### 4. Verify Share Access
```bash
rclone lsd SMB-Ramanfile:ramanfiles/cole
```

If this works, your SMB is configured!

## Configuration Details

The rclone config is stored at: `~/.config/rclone/rclone.conf`

Your SMB-Ramanfile configuration should look like:
```ini
[SMB-Ramanfile]
type = smb
host = ramanfile.local
username = ramanlab
password = [encrypted]
port = 445
domain =
```

## Using the SMB Copier in Code

### Basic Usage

```python
from fbpipe.utils.smb_copy import copy_to_smb, copy_csv_to_smb

# Copy a CSV file to standard location
copy_csv_to_smb('/path/to/file.csv')

# Copy to custom location
copy_to_smb(
    '/path/to/results/',
    'ramanfiles/cole/Figures/MyResults/'
)
```

### Advanced Usage

```python
from fbpipe.utils.smb_copy import get_smb_copier

smb = get_smb_copier()

# Copy single file
smb.copy_file(
    '/path/to/file.csv',
    'ramanfiles/cole/flyTrackingData/',
    skip_same_size=True,
    verbose=True
)

# Sync entire directory
smb.sync_directory(
    '/path/to/results/',
    'ramanfiles/cole/Figures/MyResults/'
)

# Test connection
smb.test_connection()

# Dry run (preview without copying)
smb.copy_file('/path/to/file.csv', 'ramanfiles/cole/', dry_run=True)
```

## Troubleshooting

### Error: "couldn't connect SMB"
- Check hostname: `ping ramanfile.local`
- Verify IP: `getent hosts ramanfile.local`
- Test SMB connectivity: `smbclient -L ramanfile.local -U ramanlab`

### Error: "invalid username or password"
- Re-run `rclone config` and update credentials
- Check that username is `ramanlab` and password is correct

### Error: "connection refused"
- Verify SMB is running on ramanfile.local
- Check port 445 is accessible: `nc -zv ramanfile.local 445`

### Error: "permission denied"
- Ensure you have read access to source files
- Verify write permissions on SMB share

## Rclone Common Commands

```bash
# List available remotes
rclone listremotes

# List contents of SMB share
rclone lsd SMB-Ramanfile:ramanfiles/cole

# Copy a file (dry run)
rclone copy /local/path/file.csv SMB-Ramanfile:ramanfiles/cole/path/ --dry-run

# Sync a directory
rclone sync /local/path/ SMB-Ramanfile:ramanfiles/cole/path/

# Check connection
rclone ls SMB-Ramanfile:ramanfiles -h

# View logs (verbose)
rclone copy /src SMB-Ramanfile:dest -v

# Check rclone config
rclone config dump SMB-Ramanfile
```

## Integration with Pipeline

The pipeline automatically uses SMB copying when:
1. `out_dir_smb` is defined in config.yaml
2. Analysis steps complete successfully
3. rclone SMB-Ramanfile is configured

No additional code changes needed! The pipeline will automatically:
- Copy CSV outputs to `ramanfiles/cole/flyTrackingData/`
- Copy figure outputs to `ramanfiles/cole/Figures/`
- Handle errors gracefully with logging

## Advanced Rclone Features

### Bandwidth Limiting
```bash
rclone copy /src SMB-Ramanfile:dest --bwlimit 10M
```

### Parallel Transfers
```bash
rclone copy /src SMB-Ramanfile:dest --transfers 4
```

### Skip Larger Files
```bash
rclone copy /src SMB-Ramanfile:dest --max-size 1G
```

### Check Files Match
```bash
rclone check /src SMB-Ramanfile:dest
```

## Security Notes

- Password is encrypted in `.config/rclone/rclone.conf`
- File permissions: `-rw-------` (owner only)
- Consider using a service account for production
- Rclone supports encryption at rest if needed

## See Also

- [Rclone SMB Documentation](https://rclone.org/smb/)
- [Rclone Config Reference](https://rclone.org/commands/rclone_config/)
- Project backup system: [scripts/backup_system.py](../scripts/backup_system.py)
