#!/bin/bash
# Restore everything to a fresh database
createdb publishing 2>/dev/null
pg_restore -d publishing --clean --if-exists backup.dump
echo "Done"
