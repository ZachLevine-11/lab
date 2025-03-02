#!/bin/bash

##run from within the top-level execution dir of the cromwell dir
##prints shards top-bottom from oldest to newest in terms of running directories

find . -type d | while read dir; do
    latest_file=$(find "$dir" -maxdepth 1 -type f -exec stat --format="%Y" {} + 2>/dev/null | sort -n | tail -n 1)
    if [ -n "$latest_file" ]; then
        echo "$latest_file $dir"
    fi
done | sort -n | awk '{print $2}'