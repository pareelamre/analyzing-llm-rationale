#!/bin/sh
# Verify the backup service's core operation: a consistent SQLite snapshot.
set -eu
apk add --no-cache sqlite >/dev/null 2>&1
sqlite3 /data/foresea.sqlite3 ".backup '/backups/test.sqlite3'"
echo "--- backups dir ---"
ls -la /backups
echo "--- integrity + row count of the snapshot ---"
sqlite3 /backups/test.sqlite3 "PRAGMA integrity_check;"
sqlite3 /backups/test.sqlite3 "select count(*) from entities;"
