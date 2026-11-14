#!/usr/bin/env bash
# End a GPU PC work session: stop keeping WSL alive and shut it down, freeing all of its memory.
# Run from the laptop. Start the next session with scripts/gpu-pc-up.sh.
#
# Nothing public breaks: the LLM gateway falls back to Workers AI while vLLM is off, and the
# "GPU Nightly Benchmark" task still starts WSL at 03:30 for the benchmark and shuts it down after.
set -uo pipefail

SSH_HOST="${GPU_PC_SSH:-zak-ts}"
quiet_ssh() { ssh -o ConnectTimeout=10 "$SSH_HOST" "$@" 2>&1 | tr -d '\r\0' \
  | grep -v -i 'post-quantum\|store now\|openssh.com/pq\|may need to be upgraded'; }

if [[ "${1:-}" != "--yes" ]]; then
  running=$(quiet_ssh 'wsl -d Ubuntu -u root -- docker ps --format "{{.Names}}"' | tr '\n' ' ')
  echo "This stops everything in WSL on the GPU PC: ${running:-nothing running}"
  read -r -p "Shut WSL down? [y/N] " a; [[ "$a" == [yY]* ]] || exit 1
fi

quiet_ssh 'schtasks /change /tn "WSL Keep Alive" /disable >nul & schtasks /end /tn "WSL Keep Alive" >nul & wsl --shutdown'
sleep 3
if quiet_ssh "wsl -l --running -q" | grep -qx Ubuntu; then
  echo "WSL is still running"; exit 1
fi
mem=$(quiet_ssh 'tasklist /fi "imagename eq vmmemWSL" /fo csv /nh' | grep -i vmmem | cut -d, -f5)
echo "WSL is off${mem:+ (vmmemWSL still at $mem)}. Keep-alive disabled; the nightly task still runs at 03:30."
