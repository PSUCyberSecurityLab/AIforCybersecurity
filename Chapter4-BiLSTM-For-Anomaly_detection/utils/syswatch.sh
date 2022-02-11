#!/bin/bash

GREEN='\033[0;32m'
CEND='\033[0m'

username=$(whoami)

# shellcheck disable=SC2050
while [ 1 = 1 ]; do
clear
echo ""
echo -e "${GREEN}[File System]${CEND}"; df -h | grep "\(Mounted on\| /$\)"
echo ""
echo -e "${GREEN}[Memory]${CEND}"; free -g
echo ""
# shellcheck disable=SC2153
echo -e "${GREEN}[GPU]${CEMD}"; gpustat  # pip install gpu stat
echo "" 
echo -e "${GREEN}[Top]${CEND}"; top -bn1 -d0.01 -Siu "${username}" | head -n30
read -t 30
done

# to configure top in top
# - press C to show full command
# - press W to save top configuration