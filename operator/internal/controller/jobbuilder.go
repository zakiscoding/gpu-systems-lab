package controller

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	gpuv1 "github.com/zakiscoding/gpu-systems-lab/operator/api/v1alpha1"
)

const (
	// GPUResource is the extended resource the NVIDIA device plugin advertises.
	GPUResource corev1.ResourceName = "nvidia.com/gpu"

	// LabelGpuJob is set on the Job and its pods to the owning GpuJob's name.
	LabelGpuJob = "gpu.zakiscoding.dev/gpujob"
	// AnnotationSpecHash records the GpuJob spec the Job was built from.
	AnnotationSpecHash = "gpu.zakiscoding.dev/spec-hash"

	labelManagedBy = "app.kubernetes.io/managed-by"
	managerName    = "gpujob-operator"
	containerName  = "main"
)

// BuildJob turns a GpuJob into the batch/v1 Job that runs it. It is a pure
// function (no API calls) so it is unit-tested on its own. The controller adds
// the owner reference.
//
// The pod always gets runtimeClassName (normally "nvidia") and an
// nvidia.com/gpu limit, which is what the repo's ValidatingAdmissionPolicy
// (scheduling/manifests/50-require-gpu-limit-policy.yaml) requires.
func BuildJob(gj *gpuv1.GpuJob, runtimeClass string) *batchv1.Job {
	gpus := gj.Spec.GPUs
	if gpus < 1 {
		gpus = 1 // the CRD defaults and validates this; be safe if called on an undefaulted object
	}

	res := corev1.ResourceRequirements{
		Limits:   corev1.ResourceList{GPUResource: *resource.NewQuantity(int64(gpus), resource.DecimalSI)},
		Requests: corev1.ResourceList{},
	}
	if gj.Spec.CPU != nil {
		res.Requests[corev1.ResourceCPU] = gj.Spec.CPU.DeepCopy()
	}
	if gj.Spec.Memory != nil {
		res.Requests[corev1.ResourceMemory] = gj.Spec.Memory.DeepCopy()
		res.Limits[corev1.ResourceMemory] = gj.Spec.Memory.DeepCopy()
	}
	if len(res.Requests) == 0 {
		res.Requests = nil
	}

	backoff := int32(0)
	if gj.Spec.BackoffLimit != nil {
		backoff = *gj.Spec.BackoffLimit
	}

	labels := map[string]string{
		labelManagedBy: managerName,
		LabelGpuJob:    gj.Name,
	}
	podLabels := map[string]string{LabelGpuJob: gj.Name}

	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:        gj.Name, // same name: the lookup is a Get, and a second Create can only conflict
			Namespace:   gj.Namespace,
			Labels:      labels,
			Annotations: map[string]string{AnnotationSpecHash: SpecHash(&gj.Spec)},
		},
		Spec: batchv1.JobSpec{
			BackoffLimit:            &backoff,
			ActiveDeadlineSeconds:   copyPtr(gj.Spec.ActiveDeadlineSeconds),
			TTLSecondsAfterFinished: copyPtr(gj.Spec.TTLSecondsAfterFinished),
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: podLabels},
				Spec: corev1.PodSpec{
					RestartPolicy:    corev1.RestartPolicyNever,
					RuntimeClassName: &runtimeClass,
					Containers: []corev1.Container{{
						Name:      containerName,
						Image:     gj.Spec.Image,
						Command:   append([]string(nil), gj.Spec.Command...),
						Args:      append([]string(nil), gj.Spec.Args...),
						Env:       append([]corev1.EnvVar(nil), gj.Spec.Env...),
						Resources: res,
					}},
				},
			},
		},
	}
}

// SpecHash is a short, stable hash of a GpuJob spec. It is stored on the Job
// so drift between the GpuJob and its Job is visible (see the README).
func SpecHash(spec *gpuv1.GpuJobSpec) string {
	b, _ := json.Marshal(spec) // a struct of plain fields; Marshal cannot fail
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:8])
}

func copyPtr[T any](p *T) *T {
	if p == nil {
		return nil
	}
	v := *p
	return &v
}
