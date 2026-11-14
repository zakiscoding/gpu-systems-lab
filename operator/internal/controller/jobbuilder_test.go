package controller

import (
	"testing"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"

	gpuv1 "github.com/zakiscoding/gpu-systems-lab/operator/api/v1alpha1"
)

func sampleGpuJob() *gpuv1.GpuJob {
	cpu, mem := resource.MustParse("500m"), resource.MustParse("2Gi")
	return &gpuv1.GpuJob{
		ObjectMeta: metav1.ObjectMeta{Name: "mm", Namespace: "gpu-jobs"},
		Spec: gpuv1.GpuJobSpec{
			Image:                   "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime",
			Command:                 []string{"python", "/bench/bench.py"},
			Args:                    []string{"--n", "8192"},
			Env:                     []corev1.EnvVar{{Name: "DURATION", Value: "15"}},
			GPUs:                    2,
			CPU:                     &cpu,
			Memory:                  &mem,
			BackoffLimit:            ptr.To[int32](1),
			ActiveDeadlineSeconds:   ptr.To[int64](600),
			TTLSecondsAfterFinished: ptr.To[int32](3600),
		},
	}
}

func TestBuildJobGPUAndRuntime(t *testing.T) {
	job := BuildJob(sampleGpuJob(), "nvidia")
	pod := job.Spec.Template.Spec
	if pod.RuntimeClassName == nil || *pod.RuntimeClassName != "nvidia" {
		t.Fatalf("runtimeClassName = %v, want nvidia", pod.RuntimeClassName)
	}
	if pod.RestartPolicy != corev1.RestartPolicyNever {
		t.Errorf("restartPolicy = %s, want Never", pod.RestartPolicy)
	}
	if len(pod.Containers) != 1 {
		t.Fatalf("got %d containers, want 1", len(pod.Containers))
	}
	c := pod.Containers[0]
	if q := c.Resources.Limits[GPUResource]; q.Value() != 2 {
		t.Errorf("gpu limit = %s, want 2", q.String())
	}
	if q := c.Resources.Requests[corev1.ResourceCPU]; q.String() != "500m" {
		t.Errorf("cpu request = %s, want 500m", q.String())
	}
	if q := c.Resources.Limits[corev1.ResourceMemory]; q.String() != "2Gi" {
		t.Errorf("memory limit = %s, want 2Gi", q.String())
	}
	if _, ok := c.Resources.Limits[corev1.ResourceCPU]; ok {
		t.Errorf("unexpected cpu limit")
	}
	if c.Image != "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime" || c.Command[1] != "/bench/bench.py" ||
		c.Args[1] != "8192" || c.Env[0].Value != "15" {
		t.Errorf("container fields not copied: %+v", c)
	}
}

func TestBuildJobJobFields(t *testing.T) {
	gj := sampleGpuJob()
	job := BuildJob(gj, "nvidia")
	if job.Name != "mm" || job.Namespace != "gpu-jobs" {
		t.Errorf("name = %s/%s, want gpu-jobs/mm", job.Namespace, job.Name)
	}
	if *job.Spec.BackoffLimit != 1 || *job.Spec.ActiveDeadlineSeconds != 600 || *job.Spec.TTLSecondsAfterFinished != 3600 {
		t.Errorf("job limits not copied: %+v", job.Spec)
	}
	if job.Labels[LabelGpuJob] != "mm" || job.Spec.Template.Labels[LabelGpuJob] != "mm" {
		t.Errorf("gpujob label missing: job %v pod %v", job.Labels, job.Spec.Template.Labels)
	}
	if job.Annotations[AnnotationSpecHash] != SpecHash(&gj.Spec) {
		t.Errorf("spec-hash annotation = %q", job.Annotations[AnnotationSpecHash])
	}
	// The Job must not alias the GpuJob's slices or pointers.
	gj.Spec.Command[0] = "changed"
	*gj.Spec.ActiveDeadlineSeconds = 1
	if job.Spec.Template.Spec.Containers[0].Command[0] != "python" || *job.Spec.ActiveDeadlineSeconds != 600 {
		t.Errorf("Job shares memory with the GpuJob spec")
	}
}

func TestBuildJobDefaults(t *testing.T) {
	gj := &gpuv1.GpuJob{
		ObjectMeta: metav1.ObjectMeta{Name: "x", Namespace: "default"},
		Spec:       gpuv1.GpuJobSpec{Image: "busybox:1.37"},
	}
	job := BuildJob(gj, "nvidia")
	c := job.Spec.Template.Spec.Containers[0]
	if q := c.Resources.Limits[GPUResource]; q.Value() != 1 {
		t.Errorf("gpus 0 (undefaulted) should become 1, got %s", q.String())
	}
	if c.Resources.Requests != nil {
		t.Errorf("no cpu/memory set, want nil requests, got %v", c.Resources.Requests)
	}
	if *job.Spec.BackoffLimit != 0 {
		t.Errorf("backoffLimit = %d, want 0", *job.Spec.BackoffLimit)
	}
	if job.Spec.ActiveDeadlineSeconds != nil || job.Spec.TTLSecondsAfterFinished != nil {
		t.Errorf("deadline/ttl should be unset")
	}
}

func TestBuildJobRuntimeClassFlag(t *testing.T) {
	job := BuildJob(sampleGpuJob(), "nvidia-cdi")
	if got := *job.Spec.Template.Spec.RuntimeClassName; got != "nvidia-cdi" {
		t.Errorf("runtimeClassName = %s, want nvidia-cdi", got)
	}
}

func TestSpecHash(t *testing.T) {
	a, b := sampleGpuJob(), sampleGpuJob()
	if SpecHash(&a.Spec) != SpecHash(&b.Spec) {
		t.Fatal("equal specs hash differently")
	}
	b.Spec.GPUs = 1
	if SpecHash(&a.Spec) == SpecHash(&b.Spec) {
		t.Fatal("different specs hash the same")
	}
	if len(SpecHash(&a.Spec)) != 16 {
		t.Errorf("hash length = %d, want 16", len(SpecHash(&a.Spec)))
	}
}

func TestApplyJobStatus(t *testing.T) {
	now := metav1.Now()
	cases := []struct {
		name      string
		job       batchv1.JobStatus
		pods      []corev1.Pod
		wantPhase gpuv1.GpuJobPhase
		wantSched metav1.ConditionStatus
	}{
		{"no pod", batchv1.JobStatus{}, nil, gpuv1.PhasePending, metav1.ConditionUnknown},
		{"unschedulable", batchv1.JobStatus{Active: 1, Ready: ptr.To[int32](0)},
			[]corev1.Pod{pod(corev1.PodPending, corev1.ConditionFalse, "0/1 nodes are available: 1 Insufficient nvidia.com/gpu.")},
			gpuv1.PhasePending, metav1.ConditionFalse},
		{"running", batchv1.JobStatus{Active: 1, Ready: ptr.To[int32](1), StartTime: &now},
			[]corev1.Pod{pod(corev1.PodRunning, corev1.ConditionTrue, "")}, gpuv1.PhaseRunning, metav1.ConditionTrue},
		{"pod exited, Job not final yet", batchv1.JobStatus{Active: 0, Ready: ptr.To[int32](0), StartTime: &now},
			[]corev1.Pod{pod(corev1.PodSucceeded, corev1.ConditionTrue, "")}, gpuv1.PhaseRunning, metav1.ConditionTrue},
		{"retry pod queued after a failure", batchv1.JobStatus{Active: 1, Ready: ptr.To[int32](0), Failed: 1},
			[]corev1.Pod{pod(corev1.PodFailed, corev1.ConditionTrue, ""),
				pod(corev1.PodPending, corev1.ConditionFalse, "0/1 nodes are available: 1 Insufficient nvidia.com/gpu.")},
			gpuv1.PhasePending, metav1.ConditionFalse},
		{"complete", batchv1.JobStatus{Succeeded: 1, StartTime: &now, CompletionTime: &now,
			Conditions: []batchv1.JobCondition{{Type: batchv1.JobComplete, Status: corev1.ConditionTrue}}},
			nil, gpuv1.PhaseSucceeded, ""},
		{"failed", batchv1.JobStatus{Failed: 1, StartTime: &now,
			Conditions: []batchv1.JobCondition{{Type: batchv1.JobFailed, Status: corev1.ConditionTrue,
				Reason: "BackoffLimitExceeded", LastTransitionTime: now}}},
			nil, gpuv1.PhaseFailed, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			st := &gpuv1.GpuJobStatus{}
			applyJobStatus(st, 1, &batchv1.Job{Status: tc.job}, tc.pods)
			if st.Phase != tc.wantPhase {
				t.Errorf("phase = %s, want %s (message %q)", st.Phase, tc.wantPhase, st.Message)
			}
			if tc.wantSched != "" {
				c := findCond(st, gpuv1.ConditionScheduled)
				if c == nil || c.Status != tc.wantSched {
					t.Errorf("Scheduled = %v, want %s", c, tc.wantSched)
				}
			}
			if isTerminal(st.Phase) && st.CompletionTime == nil {
				t.Errorf("terminal phase without completionTime")
			}
		})
	}
}

func pod(phase corev1.PodPhase, scheduled corev1.ConditionStatus, msg string) corev1.Pod {
	return corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "p"},
		Status: corev1.PodStatus{
			Phase:      phase,
			Conditions: []corev1.PodCondition{{Type: corev1.PodScheduled, Status: scheduled, Message: msg}},
		},
	}
}

func findCond(st *gpuv1.GpuJobStatus, typ string) *metav1.Condition {
	for i := range st.Conditions {
		if st.Conditions[i].Type == typ {
			return &st.Conditions[i]
		}
	}
	return nil
}
