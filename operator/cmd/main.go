// Command manager runs the GpuJob controller.
package main

import (
	"flag"
	"os"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	gpuv1 "github.com/zakiscoding/gpu-systems-lab/operator/api/v1alpha1"
	"github.com/zakiscoding/gpu-systems-lab/operator/internal/controller"
)

func main() {
	var metricsAddr, probeAddr, runtimeClass string
	var leaderElect bool
	flag.StringVar(&metricsAddr, "metrics-bind-address", ":8080", "Prometheus metrics address (\"0\" disables)")
	flag.StringVar(&probeAddr, "health-probe-bind-address", ":8081", "healthz/readyz address")
	flag.StringVar(&runtimeClass, "runtime-class", "nvidia", "runtimeClassName set on every GpuJob pod")
	flag.BoolVar(&leaderElect, "leader-elect", false, "use a Lease so only one replica reconciles")
	opts := zap.Options{}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()
	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&opts)))
	log := ctrl.Log.WithName("setup")

	scheme := runtime.NewScheme()
	utilruntime.Must(corev1.AddToScheme(scheme))
	utilruntime.Must(batchv1.AddToScheme(scheme))
	utilruntime.Must(gpuv1.AddToScheme(scheme))

	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme:                 scheme,
		Cache:                  controller.CacheOptions(),
		Metrics:                metricsserver.Options{BindAddress: metricsAddr},
		HealthProbeBindAddress: probeAddr,
		LeaderElection:         leaderElect,
		LeaderElectionID:       "gpujob-operator.gpu.zakiscoding.dev",
		// Hand the lease over on shutdown, so a rolling restart does not wait for it to expire.
		// Safe: main exits right after the manager stops.
		LeaderElectionReleaseOnCancel: true,
	})
	if err != nil {
		log.Error(err, "creating manager")
		os.Exit(1)
	}
	if err := (&controller.GpuJobReconciler{
		Client:       mgr.GetClient(),
		Scheme:       mgr.GetScheme(),
		Recorder:     mgr.GetEventRecorderFor("gpujob-operator"),
		RuntimeClass: runtimeClass,
	}).SetupWithManager(mgr); err != nil {
		log.Error(err, "setting up controller")
		os.Exit(1)
	}
	utilruntime.Must(mgr.AddHealthzCheck("healthz", healthz.Ping))
	utilruntime.Must(mgr.AddReadyzCheck("readyz", healthz.Ping))

	log.Info("starting manager", "runtimeClass", runtimeClass)
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		log.Error(err, "manager exited")
		os.Exit(1)
	}
}
