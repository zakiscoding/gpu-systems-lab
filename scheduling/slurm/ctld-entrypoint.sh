#!/bin/sh
# munge.key comes from the munge-key volume (generated once by setup.sh, copied to the node).
set -e
[ -s /etc/munge/munge.key ] || { echo "no /etc/munge/munge.key"; exit 1; }
chown munge:munge /etc/munge/munge.key && chmod 400 /etc/munge/munge.key
su -s /bin/sh munge -c "munged --force"
exec slurmctld -D -vv
