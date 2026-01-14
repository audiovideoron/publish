#!/bin/bash
# Backup everything - schema + data
pg_dump -Fc publishing > backup.dump
echo "Done: backup.dump"
