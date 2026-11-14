package controller

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"go.uber.org/zap/zapcore"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	nodev1 "k8s.io/api/node/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/config"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	gpuv1 "github.com/zakiscoding/gpu-systems-lab/operator/api/v1alpha1"
)

// envtest starts a real kube-apiserver and etcd (no scheduler, kubelet,
// job controller or garbage collector). The GpuJob controller runs in a
// manager against it; the tests play the part of the missing controllers by
// writing Job and Pod status themselves.

var (
	k8s    client.Client // uncached, so assertions never read a stale cache
	scheme = runtime.NewScheme()
	nsSeq  atomic.Int32
)

func TestMain(m *testing.M) {
	utilruntime.Must(corev1.AddToScheme(scheme))
	utilruntime.Must(batchv1.AddToScheme(scheme))
	utilruntime.Must(gpuv1.AddToScheme(scheme))
	utilruntime.Must(nodev1.AddToScheme(scheme))
	if os.Getenv("KUBEBUILDER_ASSETS") == "" {
		// Unit tests still run; every envtest test skips itself (see newNamespace).
		fmt.Println("KUBEBUILDER_ASSETS not set: skipping envtest tests (`make test` sets it up)")
		os.Exit(m.Run())
	}
	ctrl.SetLogger(zap.New(zap.WriteTo(os.Stderr), zap.UseDevMode(false), zap.Level(zapLevel())))

	env := &envtest.Environment{
		CRDDirectoryPaths:     []string{filepath.Join("..", "..", "config", "crd", "bases")},
		ErrorIfCRDPathMissing: true,
	}
	cfg, err := env.Start()
	if err != nil {
		fmt.Println("envtest start:", err)
		os.Exit(1)
	}
	k8s, err = client.New(cfg, client.Options{Scheme: scheme})
	if err != nil {
		panic(err)
	}

	// RuntimeClass nvidia as on k3s, so GpuJob pods pass the API server's RuntimeClass admission.
	if err := k8s.Create(context.Background(), &nodev1.RuntimeClass{
		ObjectMeta: metav1.ObjectMeta{Name: "nvidia"}, Handler: "nvidia",
	}); err != nil {
		panic(err)
	}

	mgr, err := ctrl.NewManager(cfg, ctrl.Options{
		Scheme:     scheme,
		Cache:      CacheOptions(),
		Metrics:    metricsserver.Options{BindAddress: "0"},
		Controller: config.Controller{SkipNameValidation: ptr.To(true)},
	})
	if err != nil {
		panic(err)
	}
	if err := (&GpuJobReconciler{
		Client: mgr.GetClient(), Scheme: scheme, RuntimeClass: "nvidia",
		Recorder: mgr.GetEventRecorderFor("gpujob-operator"),
	}).SetupWithManager(mgr); err != nil {
		panic(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		if err := mgr.Start(ctx); err != nil {
			panic(err)
		}
	}()

	code := m.Run()
	cancel()
	if err := env.Stop(); err != nil {
		fmt.Println("envtest stop:", err)
	}
	os.Exit(code)
}

func zapLevel() zapcore.Level {
	if os.Getenv("TEST_VERBOSE_LOGS") != "" {
		return zapcore.InfoLevel
	}
	return zapcore.ErrorLevel
}

// newNamespace gives each test its own namespace (with a default
// ServiceAccount, which pod admission wants and envtest does not create).
func newNamespace(t *testing.T) string {
	t.Helper()
	if k8s == nil {
		t.Skip("envtest not available (KUBEBUILDER_ASSETS unset)")
	}
	name := fmt.Sprintf("t%d", nsSeq.Add(1))
	ctx := context.Background()
	must(t, k8s.Create(ctx, &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: name}}))
	must(t, k8s.Create(ctx, &corev1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: "default", Namespace: name}}))
	return name
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

// eventually polls cond every 50 ms for up to 10 s.
func eventually(t *testing.T, what string, cond func() (bool, error)) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	var lastErr error
	for time.Now().Before(deadline) {
		ok, err := cond()
		if ok {
			return
		}
		lastErr = err
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s (last error: %v)", what, lastErr)
}
