# Run inside the golang container: ./hack/in-docker.sh make <target>
# (the laptop has no Go toolchain). Works the same on any machine with Go 1.26.

CONTROLLER_GEN_VERSION ?= v0.23.0
SETUP_ENVTEST_VERSION  ?= release-0.25
# Matches the k3s node (v1.36.5+k3s1).
ENVTEST_K8S_VERSION    ?= 1.36

LOCALBIN       ?= $(CURDIR)/bin
CONTROLLER_GEN ?= $(LOCALBIN)/controller-gen
SETUP_ENVTEST  ?= $(LOCALBIN)/setup-envtest
# Where kube-apiserver/etcd are unpacked. hack/in-docker.sh points this at a Docker
# volume: run from a Windows bind mount, kube-apiserver timed out on start.
ENVTEST_DIR    ?= $(LOCALBIN)

.PHONY: all generate fmt vet unit test build
all: generate fmt vet test build

$(CONTROLLER_GEN):
	GOBIN=$(LOCALBIN) go install sigs.k8s.io/controller-tools/cmd/controller-gen@$(CONTROLLER_GEN_VERSION)

$(SETUP_ENVTEST):
	GOBIN=$(LOCALBIN) go install sigs.k8s.io/controller-runtime/tools/setup-envtest@$(SETUP_ENVTEST_VERSION)

## generate: deepcopy code, CRD YAML (config/crd/bases) and RBAC (config/rbac/role.yaml)
generate: $(CONTROLLER_GEN)
	$(CONTROLLER_GEN) object paths=./api/...
	$(CONTROLLER_GEN) crd rbac:roleName=gpujob-operator paths=./... \
		output:crd:artifacts:config=config/crd/bases output:rbac:artifacts:config=config/rbac

fmt:
	gofmt -l -w api cmd internal

vet:
	go vet ./...

## unit: Job builder and status mapping only (no API server needed)
unit:
	go test ./internal/controller/ -run 'TestBuildJob|TestSpecHash|TestApplyJobStatus' -count=1 -v

## test: unit + envtest controller tests against a real kube-apiserver and etcd
test: $(SETUP_ENVTEST)
	KUBEBUILDER_ASSETS="$$($(SETUP_ENVTEST) use $(ENVTEST_K8S_VERSION) --bin-dir $(ENVTEST_DIR) -p path)" \
		go test ./... -count=1 -v

build:
	CGO_ENABLED=0 go build -o $(LOCALBIN)/manager ./cmd
