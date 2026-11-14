// Package controller reconciles GpuJobs into batch/v1 Jobs.
package controller

import (
	"context"
	"fmt"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	gpuv1 "github.com/zakiscoding/gpu-systems-lab/operator/api/v1alpha1"
)

// Reasons used in conditions and events.
const (
	ReasonCreated       = "Created"
	ReasonRecreated     = "Recreated"
	ReasonNameConflict  = "NameConflict"
	ReasonSpecDrift     = "SpecDrift"
	ReasonUnschedulable = "Unschedulable"
	ReasonBound         = "Bound"
	ReasonNoPod         = "NoPod"
	ReasonSucceeded     = "Succeeded"
	ReasonFailed        = "Failed"
	ReasonJobGone       = "JobDeletedAfterFinish"

	// ConditionSpecInSync is False when the Job's spec-hash annotation does not
	// match the GpuJob spec (see the README: drift is reported, never "fixed").
	ConditionSpecInSync = "SpecInSync"
)

// GpuJobReconciler turns each GpuJob into exactly one batch/v1 Job and mirrors
// the Job's progress back into GpuJob.status.
type GpuJobReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder
	// RuntimeClass is set on every pod; "nvidia" on the k3s node.
	RuntimeClass string
}

// +kubebuilder:rbac:groups=gpu.zakiscoding.dev,resources=gpujobs,verbs=get;list;watch
// +kubebuilder:rbac:groups=gpu.zakiscoding.dev,resources=gpujobs/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=gpu.zakiscoding.dev,resources=gpujobs/finalizers,verbs=update
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create
// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list;watch
// +kubebuilder:rbac:groups="",resources=events,verbs=create;patch
// +kubebuilder:rbac:groups=events.k8s.io,resources=events,verbs=create;patch

// Reconcile is level-triggered: it reads the GpuJob, its Job and the Job's
// pods, and converges. Running it any number of times on the same state makes
// the same single Job and writes the same status.
func (r *GpuJobReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := logf.FromContext(ctx)

	var gj gpuv1.GpuJob
	if err := r.Get(ctx, req.NamespacedName, &gj); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err) // deleted: the Job goes with it (owner reference)
	}
	if !gj.DeletionTimestamp.IsZero() {
		return ctrl.Result{}, nil
	}
	status := gj.Status.DeepCopy()
	status.ObservedGeneration = gj.Generation

	var job batchv1.Job
	err := r.Get(ctx, types.NamespacedName{Namespace: gj.Namespace, Name: gj.Name}, &job)
	switch {
	case apierrors.IsNotFound(err):
		if isTerminal(status.Phase) {
			// The Job finished and was removed afterwards (ttlSecondsAfterFinished, or by
			// hand). Re-running a finished job is never what the user wants: keep the result.
			setCond(status, gj.Generation, gpuv1.ConditionJobCreated, metav1.ConditionFalse, ReasonJobGone,
				"the Job was deleted after it finished; final status kept")
			return ctrl.Result{}, r.writeStatus(ctx, &gj, status)
		}
		reason := ReasonCreated
		if status.JobName != "" {
			reason = ReasonRecreated // it existed before and vanished while unfinished
		}
		job = *BuildJob(&gj, r.RuntimeClass)
		if err := controllerutil.SetControllerReference(&gj, &job, r.Scheme); err != nil {
			return ctrl.Result{}, err
		}
		if err := r.Create(ctx, &job); err != nil {
			if apierrors.IsAlreadyExists(err) {
				// The cache has not seen our earlier Create yet. Try again; the next pass will find it.
				return ctrl.Result{RequeueAfter: time.Second}, nil
			}
			return ctrl.Result{}, err
		}
		log.Info("created Job", "job", job.Name, "reason", reason)
		r.Recorder.Eventf(&gj, corev1.EventTypeNormal, reason, "%s Job %s", reason, job.Name)
		status.JobName = job.Name
		status.Phase = gpuv1.PhasePending
		status.StartTime, status.CompletionTime = nil, nil
		status.Message = "Job created"
		setCond(status, gj.Generation, gpuv1.ConditionJobCreated, metav1.ConditionTrue, reason, "Job "+job.Name)
		return ctrl.Result{}, r.writeStatus(ctx, &gj, status)
	case err != nil:
		return ctrl.Result{}, err
	}

	if !metav1.IsControlledBy(&job, &gj) {
		// A Job with our name that we do not own. Never adopt or delete someone else's Job.
		status.Phase = gpuv1.PhaseFailed
		status.Message = fmt.Sprintf("a Job named %s already exists and is not owned by this GpuJob", job.Name)
		setCond(status, gj.Generation, gpuv1.ConditionJobCreated, metav1.ConditionFalse, ReasonNameConflict, status.Message)
		r.Recorder.Event(&gj, corev1.EventTypeWarning, ReasonNameConflict, status.Message)
		return ctrl.Result{}, r.writeStatus(ctx, &gj, status)
	}
	status.JobName = job.Name
	if !meta.IsStatusConditionTrue(status.Conditions, gpuv1.ConditionJobCreated) { // keep a "Recreated" reason
		setCond(status, gj.Generation, gpuv1.ConditionJobCreated, metav1.ConditionTrue, ReasonCreated, "Job "+job.Name)
	}

	// Drift: the spec is immutable (CRD rule), so a mismatch means the operator's
	// Job template changed (upgrade) or the rule was bypassed. Job pod templates
	// are immutable too, and replacing a running GPU job would throw away its
	// work, so we only report it.
	if want := SpecHash(&gj.Spec); job.Annotations[AnnotationSpecHash] != want {
		setCond(status, gj.Generation, ConditionSpecInSync, metav1.ConditionFalse, ReasonSpecDrift,
			fmt.Sprintf("Job was built from spec %s, GpuJob spec is %s; delete and re-create the GpuJob to apply it",
				job.Annotations[AnnotationSpecHash], want))
	} else {
		setCond(status, gj.Generation, ConditionSpecInSync, metav1.ConditionTrue, ReasonCreated, "")
	}

	var pods corev1.PodList
	if err := r.List(ctx, &pods, client.InNamespace(gj.Namespace), client.MatchingLabels{LabelGpuJob: gj.Name}); err != nil {
		return ctrl.Result{}, err
	}
	applyJobStatus(status, gj.Generation, &job, pods.Items)
	return ctrl.Result{}, r.writeStatus(ctx, &gj, status)
}

// applyJobStatus maps the Job (and its pods) onto the GpuJob status.
func applyJobStatus(status *gpuv1.GpuJobStatus, gen int64, job *batchv1.Job, pods []corev1.Pod) {
	status.StartTime = job.Status.StartTime

	if c := jobCondition(job, batchv1.JobComplete); c != nil {
		status.Phase = gpuv1.PhaseSucceeded
		status.CompletionTime = job.Status.CompletionTime
		if status.CompletionTime == nil {
			status.CompletionTime = &c.LastTransitionTime
		}
		status.Message = "Job completed"
		setCond(status, gen, gpuv1.ConditionFinished, metav1.ConditionTrue, ReasonSucceeded, status.Message)
		return
	}
	if c := jobCondition(job, batchv1.JobFailed); c != nil {
		status.Phase = gpuv1.PhaseFailed
		status.CompletionTime = &c.LastTransitionTime // the Job sets completionTime only on success
		status.Message = fmt.Sprintf("%s: %s", c.Reason, c.Message)
		setCond(status, gen, gpuv1.ConditionFinished, metav1.ConditionTrue, ReasonFailed, status.Message)
		return
	}
	setCond(status, gen, gpuv1.ConditionFinished, metav1.ConditionFalse, "Active", "")
	status.CompletionTime = nil

	running := job.Status.Ready != nil && *job.Status.Ready > 0
	sched, schedReason, schedMsg := metav1.ConditionUnknown, ReasonNoPod, "no pod yet"
	waiting := ""
	unstarted, finished := false, false
	for i := range pods {
		switch pods[i].Status.Phase {
		case corev1.PodRunning:
			running = true
		case corev1.PodSucceeded, corev1.PodFailed:
			finished = true
		default:
			unstarted = true
		}
	}
	for i := range pods {
		p := &pods[i]
		exited := p.Status.Phase == corev1.PodSucceeded || p.Status.Phase == corev1.PodFailed
		if exited && (running || unstarted) {
			continue // a retry pod is what matters for Scheduled, not the one that already failed
		}
		for _, pc := range p.Status.Conditions {
			if pc.Type != corev1.PodScheduled {
				continue
			}
			if pc.Status == corev1.ConditionTrue {
				sched, schedReason, schedMsg = metav1.ConditionTrue, ReasonBound, "pod "+p.Name+" bound to "+p.Spec.NodeName
			} else if sched != metav1.ConditionTrue {
				sched, schedReason, schedMsg = metav1.ConditionFalse, ReasonUnschedulable, pc.Message
			}
		}
		for _, cs := range p.Status.ContainerStatuses {
			if cs.State.Waiting != nil && cs.State.Waiting.Reason != "" {
				waiting = cs.State.Waiting.Reason
			}
		}
	}
	setCond(status, gen, gpuv1.ConditionScheduled, sched, schedReason, schedMsg)

	switch {
	case running:
		status.Phase = gpuv1.PhaseRunning
		status.Message = "pod running"
	case finished && !unstarted:
		// The pod has exited but the Job controller has not written Complete/Failed
		// yet (a second or so). Stay Running instead of flickering back to Pending.
		status.Phase = gpuv1.PhaseRunning
		status.Message = "pod finished, waiting for the Job's final status"
	case sched == metav1.ConditionFalse:
		status.Phase = gpuv1.PhasePending
		status.Message = "queued: " + schedMsg
	case waiting != "":
		status.Phase = gpuv1.PhasePending
		status.Message = waiting
	default:
		status.Phase = gpuv1.PhasePending
		status.Message = "waiting for pod"
	}
}

// writeStatus updates the status subresource, but only when something changed,
// so a no-op reconcile is also a no-op on the API server.
func (r *GpuJobReconciler) writeStatus(ctx context.Context, gj *gpuv1.GpuJob, status *gpuv1.GpuJobStatus) error {
	if equality.Semantic.DeepEqual(&gj.Status, status) {
		return nil
	}
	orig := gj.DeepCopy()
	gj.Status = *status
	// Merge patch on the status subresource: only this controller writes status, so no
	// resourceVersion precondition is needed, and spec/metadata writers never conflict with it.
	return r.Status().Patch(ctx, gj, client.MergeFrom(orig))
}

func setCond(status *gpuv1.GpuJobStatus, gen int64, typ string, s metav1.ConditionStatus, reason, msg string) {
	meta.SetStatusCondition(&status.Conditions, metav1.Condition{
		Type: typ, Status: s, Reason: reason, Message: msg, ObservedGeneration: gen,
	})
}

func jobCondition(job *batchv1.Job, t batchv1.JobConditionType) *batchv1.JobCondition {
	for i := range job.Status.Conditions {
		if c := &job.Status.Conditions[i]; c.Type == t && c.Status == corev1.ConditionTrue {
			return c
		}
	}
	return nil
}

func isTerminal(p gpuv1.GpuJobPhase) bool {
	return p == gpuv1.PhaseSucceeded || p == gpuv1.PhaseFailed
}

// podToGpuJob maps a pod to the GpuJob named in its label.
func podToGpuJob(_ context.Context, o client.Object) []reconcile.Request {
	name, ok := o.GetLabels()[LabelGpuJob]
	if !ok {
		return nil
	}
	return []reconcile.Request{{NamespacedName: types.NamespacedName{Namespace: o.GetNamespace(), Name: name}}}
}

// CacheOptions limits the pod informer to GpuJob pods, so the operator does
// not cache every pod in the cluster.
func CacheOptions() cache.Options {
	req, _ := labels.NewRequirement(LabelGpuJob, selection.Exists, nil)
	return cache.Options{ByObject: map[client.Object]cache.ByObject{
		&corev1.Pod{}: {Label: labels.NewSelector().Add(*req)},
	}}
}

// SetupWithManager registers the controller: GpuJobs, the Jobs they own, and
// the pods of those Jobs (for the Scheduled condition and the Running phase).
func (r *GpuJobReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&gpuv1.GpuJob{}).
		Owns(&batchv1.Job{}).
		Watches(&corev1.Pod{}, handler.EnqueueRequestsFromMapFunc(podToGpuJob)).
		Named("gpujob").
		Complete(r)
}
