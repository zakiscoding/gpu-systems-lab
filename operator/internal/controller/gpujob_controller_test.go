package controller

import (
	"context"
	"strings"
	"testing"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/retry"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	gpuv1 "github.com/zakiscoding/gpu-systems-lab/operator/api/v1alpha1"
)

func createGpuJob(t *testing.T, ns, name string) *gpuv1.GpuJob {
	t.Helper()
	gj := &gpuv1.GpuJob{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
		Spec: gpuv1.GpuJobSpec{
			Image:   "busybox:1.37",
			Command: []string{"sh", "-c", "sleep 1"},
		},
	}
	must(t, k8s.Create(context.Background(), gj))
	return gj
}

func getJob(ns, name string) (*batchv1.Job, error) {
	var job batchv1.Job
	err := k8s.Get(context.Background(), types.NamespacedName{Namespace: ns, Name: name}, &job)
	return &job, err
}

func getGpuJob(t *testing.T, ns, name string) *gpuv1.GpuJob {
	t.Helper()
	var gj gpuv1.GpuJob
	must(t, k8s.Get(context.Background(), types.NamespacedName{Namespace: ns, Name: name}, &gj))
	return &gj
}

func waitJob(t *testing.T, ns, name string) *batchv1.Job {
	t.Helper()
	var job *batchv1.Job
	eventually(t, "Job "+name, func() (bool, error) {
		var err error
		job, err = getJob(ns, name)
		return err == nil, err
	})
	return job
}

func waitPhase(t *testing.T, ns, name string, phase gpuv1.GpuJobPhase) *gpuv1.GpuJob {
	t.Helper()
	var gj *gpuv1.GpuJob
	eventually(t, "phase "+string(phase), func() (bool, error) {
		gj = getGpuJob(t, ns, name)
		return gj.Status.Phase == phase, nil
	})
	return gj
}

func TestCreatesJobWithGPULimitRuntimeClassAndOwner(t *testing.T) {
	ns := newNamespace(t)
	gj := createGpuJob(t, ns, "mm")
	job := waitJob(t, ns, "mm")

	pod := job.Spec.Template.Spec
	if pod.RuntimeClassName == nil || *pod.RuntimeClassName != "nvidia" {
		t.Errorf("runtimeClassName = %v, want nvidia", pod.RuntimeClassName)
	}
	if q := pod.Containers[0].Resources.Limits[GPUResource]; q.Value() != 1 {
		t.Errorf("nvidia.com/gpu limit = %s, want 1 (CRD default)", q.String())
	}
	owner := metav1.GetControllerOf(job)
	if owner == nil || owner.UID != getGpuJob(t, ns, "mm").UID || owner.Kind != "GpuJob" {
		t.Fatalf("controller owner ref = %+v, want GpuJob %s", owner, gj.Name)
	}
	if owner.BlockOwnerDeletion == nil || !*owner.BlockOwnerDeletion {
		t.Errorf("blockOwnerDeletion not set")
	}

	got := waitPhase(t, ns, "mm", gpuv1.PhasePending)
	if got.Status.JobName != "mm" || got.Status.ObservedGeneration != got.Generation {
		t.Errorf("status = %+v", got.Status)
	}
	if !meta.IsStatusConditionTrue(got.Status.Conditions, gpuv1.ConditionJobCreated) {
		t.Errorf("JobCreated not True: %+v", got.Status.Conditions)
	}
}

func TestReconcileIsIdempotent(t *testing.T) {
	ns := newNamespace(t)
	createGpuJob(t, ns, "idem")
	job := waitJob(t, ns, "idem")
	waitPhase(t, ns, "idem", gpuv1.PhasePending)
	// Let the manager settle (it reconciles again when it sees its own Job and status write).
	time.Sleep(500 * time.Millisecond)
	gjBefore := getGpuJob(t, ns, "idem")

	// A second reconciler with an uncached client: no cache lag can hide a duplicate.
	r := &GpuJobReconciler{Client: k8s, Scheme: scheme, RuntimeClass: "nvidia", Recorder: record.NewFakeRecorder(10)}
	req := ctrl.Request{NamespacedName: types.NamespacedName{Namespace: ns, Name: "idem"}}
	for i := 0; i < 3; i++ {
		if _, err := r.Reconcile(context.Background(), req); err != nil {
			t.Fatalf("reconcile %d: %v", i, err)
		}
	}

	var jobs batchv1.JobList
	must(t, k8s.List(context.Background(), &jobs, client.InNamespace(ns)))
	if len(jobs.Items) != 1 {
		t.Fatalf("got %d Jobs after 3 extra reconciles, want 1", len(jobs.Items))
	}
	after, _ := getJob(ns, "idem")
	if after.ResourceVersion != job.ResourceVersion {
		t.Errorf("Job was rewritten: resourceVersion %s -> %s", job.ResourceVersion, after.ResourceVersion)
	}
	if gjAfter := getGpuJob(t, ns, "idem"); gjAfter.ResourceVersion != gjBefore.ResourceVersion {
		t.Errorf("no-op reconcile wrote the GpuJob: resourceVersion %s -> %s", gjBefore.ResourceVersion, gjAfter.ResourceVersion)
	}
}

// finishJob writes the status the job controller would write. The API server
// validates terminal Job status, so it needs the same conditions the real
// controller sets (SuccessCriteriaMet before Complete, FailureTarget before Failed).
func finishJob(t *testing.T, ns, name string, succeed bool) {
	t.Helper()
	job, err := getJob(ns, name)
	must(t, err)
	now := metav1.Now()
	start := metav1.NewTime(now.Add(-5 * time.Second))
	job.Status.StartTime = &start
	job.Status.Active = 0
	job.Status.Ready = new(int32)
	if succeed {
		job.Status.Succeeded = 1
		job.Status.CompletionTime = &now
		job.Status.Conditions = []batchv1.JobCondition{
			{Type: batchv1.JobSuccessCriteriaMet, Status: corev1.ConditionTrue, Reason: "CompletionsReached", LastTransitionTime: now, LastProbeTime: now},
			{Type: batchv1.JobComplete, Status: corev1.ConditionTrue, Reason: "CompletionsReached", LastTransitionTime: now, LastProbeTime: now},
		}
	} else {
		job.Status.Failed = 1
		job.Status.Conditions = []batchv1.JobCondition{
			{Type: batchv1.JobFailureTarget, Status: corev1.ConditionTrue, Reason: "BackoffLimitExceeded", Message: "Job has reached the specified backoff limit", LastTransitionTime: now, LastProbeTime: now},
			{Type: batchv1.JobFailed, Status: corev1.ConditionTrue, Reason: "BackoffLimitExceeded", Message: "Job has reached the specified backoff limit", LastTransitionTime: now, LastProbeTime: now},
		}
	}
	must(t, k8s.Status().Update(context.Background(), job))
}

func TestStatusSucceeded(t *testing.T) {
	ns := newNamespace(t)
	createGpuJob(t, ns, "ok")
	waitJob(t, ns, "ok")
	finishJob(t, ns, "ok", true)

	gj := waitPhase(t, ns, "ok", gpuv1.PhaseSucceeded)
	if gj.Status.StartTime == nil || gj.Status.CompletionTime == nil {
		t.Errorf("start/completion time missing: %+v", gj.Status)
	}
	c := meta.FindStatusCondition(gj.Status.Conditions, gpuv1.ConditionFinished)
	if c == nil || c.Status != metav1.ConditionTrue || c.Reason != ReasonSucceeded {
		t.Errorf("Finished condition = %+v", c)
	}
}

func TestStatusFailed(t *testing.T) {
	ns := newNamespace(t)
	createGpuJob(t, ns, "bad")
	waitJob(t, ns, "bad")
	finishJob(t, ns, "bad", false)

	gj := waitPhase(t, ns, "bad", gpuv1.PhaseFailed)
	if gj.Status.CompletionTime == nil || !strings.Contains(gj.Status.Message, "BackoffLimitExceeded") {
		t.Errorf("status = %+v", gj.Status)
	}
	c := meta.FindStatusCondition(gj.Status.Conditions, gpuv1.ConditionFinished)
	if c == nil || c.Reason != ReasonFailed {
		t.Errorf("Finished condition = %+v", c)
	}
}

// TestPendingQueuedThenRunning: an unschedulable pod (no free GPU) keeps the
// GpuJob Pending with Scheduled=False; once bound and running it is Running.
func TestPendingQueuedThenRunning(t *testing.T) {
	ns := newNamespace(t)
	createGpuJob(t, ns, "q")
	job := waitJob(t, ns, "q")

	tmpl := job.Spec.Template
	p := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "q-abcde", Namespace: ns, Labels: tmpl.Labels},
		Spec:       tmpl.Spec,
	}
	must(t, k8s.Create(context.Background(), p))
	p.Status = corev1.PodStatus{Phase: corev1.PodPending, Conditions: []corev1.PodCondition{{
		Type: corev1.PodScheduled, Status: corev1.ConditionFalse, Reason: "Unschedulable",
		Message: "0/1 nodes are available: 1 Insufficient nvidia.com/gpu.", LastTransitionTime: metav1.Now(),
	}}}
	must(t, k8s.Status().Update(context.Background(), p))

	eventually(t, "Scheduled=False", func() (bool, error) {
		gj := getGpuJob(t, ns, "q")
		c := meta.FindStatusCondition(gj.Status.Conditions, gpuv1.ConditionScheduled)
		return c != nil && c.Status == metav1.ConditionFalse && gj.Status.Phase == gpuv1.PhasePending &&
			strings.Contains(gj.Status.Message, "Insufficient nvidia.com/gpu"), nil
	})

	p.Status = corev1.PodStatus{Phase: corev1.PodRunning, Conditions: []corev1.PodCondition{{
		Type: corev1.PodScheduled, Status: corev1.ConditionTrue, LastTransitionTime: metav1.Now(),
	}}}
	must(t, k8s.Status().Update(context.Background(), p))
	gj := waitPhase(t, ns, "q", gpuv1.PhaseRunning)
	if !meta.IsStatusConditionTrue(gj.Status.Conditions, gpuv1.ConditionScheduled) {
		t.Errorf("Scheduled not True: %+v", gj.Status.Conditions)
	}
}

func TestJobDeletedWhileUnfinishedIsRecreated(t *testing.T) {
	ns := newNamespace(t)
	createGpuJob(t, ns, "re")
	old := waitJob(t, ns, "re")
	waitPhase(t, ns, "re", gpuv1.PhasePending)

	must(t, k8s.Delete(context.Background(), old, client.PropagationPolicy(metav1.DeletePropagationBackground)))
	eventually(t, "a new Job", func() (bool, error) {
		j, err := getJob(ns, "re")
		return err == nil && j.UID != old.UID, err
	})
	eventually(t, "JobCreated reason Recreated", func() (bool, error) {
		c := meta.FindStatusCondition(getGpuJob(t, ns, "re").Status.Conditions, gpuv1.ConditionJobCreated)
		return c != nil && c.Reason == ReasonRecreated, nil
	})
}

func TestJobDeletedAfterFinishIsNotRecreated(t *testing.T) {
	ns := newNamespace(t)
	createGpuJob(t, ns, "ttl")
	job := waitJob(t, ns, "ttl")
	finishJob(t, ns, "ttl", true)
	waitPhase(t, ns, "ttl", gpuv1.PhaseSucceeded)

	// What ttlSecondsAfterFinished does.
	must(t, k8s.Delete(context.Background(), job, client.PropagationPolicy(metav1.DeletePropagationBackground)))
	eventually(t, "JobCreated=False JobDeletedAfterFinish", func() (bool, error) {
		c := meta.FindStatusCondition(getGpuJob(t, ns, "ttl").Status.Conditions, gpuv1.ConditionJobCreated)
		return c != nil && c.Status == metav1.ConditionFalse && c.Reason == ReasonJobGone, nil
	})
	time.Sleep(time.Second)
	if _, err := getJob(ns, "ttl"); !apierrors.IsNotFound(err) {
		t.Fatalf("finished GpuJob's Job was recreated (err=%v)", err)
	}
	if gj := getGpuJob(t, ns, "ttl"); gj.Status.Phase != gpuv1.PhaseSucceeded {
		t.Errorf("phase = %s, want Succeeded kept", gj.Status.Phase)
	}
}

func TestNameConflictWithForeignJob(t *testing.T) {
	ns := newNamespace(t)
	foreign := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{Name: "taken", Namespace: ns},
		Spec: batchv1.JobSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyNever,
			Containers:    []corev1.Container{{Name: "c", Image: "busybox:1.37"}},
		}}},
	}
	must(t, k8s.Create(context.Background(), foreign))
	createGpuJob(t, ns, "taken")

	gj := waitPhase(t, ns, "taken", gpuv1.PhaseFailed)
	c := meta.FindStatusCondition(gj.Status.Conditions, gpuv1.ConditionJobCreated)
	if c == nil || c.Reason != ReasonNameConflict {
		t.Errorf("JobCreated = %+v, want NameConflict", c)
	}
	j, _ := getJob(ns, "taken")
	if len(j.OwnerReferences) != 0 {
		t.Errorf("foreign Job was adopted: %+v", j.OwnerReferences)
	}
}

func TestSpecDriftIsReported(t *testing.T) {
	ns := newNamespace(t)
	createGpuJob(t, ns, "drift")
	job := waitJob(t, ns, "drift")
	job.Annotations[AnnotationSpecHash] = "0000000000000000"
	must(t, k8s.Update(context.Background(), job))

	eventually(t, "SpecInSync=False", func() (bool, error) {
		c := meta.FindStatusCondition(getGpuJob(t, ns, "drift").Status.Conditions, ConditionSpecInSync)
		return c != nil && c.Status == metav1.ConditionFalse && c.Reason == ReasonSpecDrift, nil
	})
	if j, _ := getJob(ns, "drift"); j.UID != job.UID {
		t.Errorf("drifted Job was replaced; it should only be reported")
	}
}

func unstructuredGpuJob(ns, name string, spec map[string]any) *unstructured.Unstructured {
	u := &unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "gpu.zakiscoding.dev/v1alpha1",
		"kind":       "GpuJob",
		"metadata":   map[string]any{"name": name, "namespace": ns},
		"spec":       spec,
	}}
	return u
}

func TestCRDValidation(t *testing.T) {
	ns := newNamespace(t)
	ctx := context.Background()
	reject := []struct {
		name string
		spec map[string]any
		want string
	}{
		{"gpus-zero", map[string]any{"image": "busybox:1.37", "gpus": int64(0)}, "spec.gpus"},
		{"gpus-nine", map[string]any{"image": "busybox:1.37", "gpus": int64(9)}, "spec.gpus"},
		{"no-image", map[string]any{"gpus": int64(1)}, "spec.image"},
		{"empty-image", map[string]any{"image": ""}, "spec.image"},
		{"negative-backoff", map[string]any{"image": "busybox:1.37", "backoffLimit": int64(-1)}, "spec.backoffLimit"},
		{"zero-deadline", map[string]any{"image": "busybox:1.37", "activeDeadlineSeconds": int64(0)}, "spec.activeDeadlineSeconds"},
	}
	for _, tc := range reject {
		t.Run(tc.name, func(t *testing.T) {
			err := k8s.Create(ctx, unstructuredGpuJob(ns, tc.name, tc.spec))
			if !apierrors.IsInvalid(err) || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("want Invalid on %s, got %v", tc.want, err)
			}
		})
	}

	t.Run("defaults", func(t *testing.T) {
		must(t, k8s.Create(ctx, unstructuredGpuJob(ns, "defaults", map[string]any{"image": "busybox:1.37"})))
		gj := getGpuJob(t, ns, "defaults")
		if gj.Spec.GPUs != 1 || gj.Spec.BackoffLimit == nil || *gj.Spec.BackoffLimit != 0 {
			t.Errorf("defaults not applied: gpus=%d backoffLimit=%v", gj.Spec.GPUs, gj.Spec.BackoffLimit)
		}
	})

	t.Run("spec-immutable", func(t *testing.T) {
		// Retry on conflict: the controller writes status concurrently.
		err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
			gj := getGpuJob(t, ns, "defaults")
			gj.Spec.GPUs = 2
			return k8s.Update(ctx, gj)
		})
		if !apierrors.IsInvalid(err) || !strings.Contains(err.Error(), "spec is immutable") {
			t.Errorf("want immutable-spec rejection, got %v", err)
		}
	})
}
