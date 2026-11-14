# kind + kubectl + docker CLI, for driving a kind cluster from a container
# (the laptop has none of them installed). Built by run-demo.sh.
FROM golang:1.26 AS tools
ARG KIND_VERSION=v0.33.0
ARG KUBECTL_VERSION=v1.36.5
RUN CGO_ENABLED=0 GOBIN=/out go install sigs.k8s.io/kind@${KIND_VERSION} \
 && curl -fsSLo /out/kubectl https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl \
 && chmod +x /out/kubectl

FROM docker:29-cli
RUN apk add --no-cache bash coreutils
COPY --from=tools /out/ /usr/local/bin/
