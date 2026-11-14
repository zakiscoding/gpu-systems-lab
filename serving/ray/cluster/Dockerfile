# A Ray node that joins the tailnet itself. Ray nodes connect to each other on many ports in both
# directions, which NAT (WSL2) and Docker bridges break. Running tailscaled inside the container
# gives every node its own routable 100.x address, so Ray sees a flat network.
#   BASE=rayproject/ray:2.60.0-py313-cu128   (GPU PC head; py3.13 to match the iMac, whose Ray runs natively)
#   (the iMac runs Ray natively: its Colima containers have no outbound network)
ARG BASE
FROM ${BASE}
ARG TS_VERSION=1.104.7
USER root
RUN ARCH=$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/') \
 && wget -qO- https://pkgs.tailscale.com/stable/tailscale_${TS_VERSION}_${ARCH}.tgz | tar xz -C /tmp \
 && install /tmp/tailscale_${TS_VERSION}_${ARCH}/tailscale /tmp/tailscale_${TS_VERSION}_${ARCH}/tailscaled /usr/local/bin/ \
 && rm -rf /tmp/tailscale_*
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
COPY ts-ray.sh /usr/local/bin/ts-ray.sh
ENTRYPOINT ["bash", "/usr/local/bin/ts-ray.sh"]
