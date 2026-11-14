package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// GpuJobSpec is what the user asks for: one container that needs `gpus` whole
// nvidia.com/gpu units. The controller turns it into a batch/v1 Job.
//
// The whole spec is immutable (CEL rule on GpuJob.spec). Almost every field
// ends up in the Job's pod template, which Kubernetes itself does not let you
// change after creation, so "edit the GpuJob" could only mean "kill the
// running pod and start over". That should be an explicit delete + create.
type GpuJobSpec struct {
	// Image is the container image to run.
	// +kubebuilder:validation:MinLength=1
	Image string `json:"image"`

	// Command overrides the image ENTRYPOINT.
	// +optional
	Command []string `json:"command,omitempty"`

	// Args overrides the image CMD.
	// +optional
	Args []string `json:"args,omitempty"`

	// Env is passed through to the container unchanged.
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`

	// GPUs is the number of nvidia.com/gpu units requested (limit == request).
	// On a time-sliced node one unit is a time slice of the card, not a card.
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=8
	// +optional
	GPUs int32 `json:"gpus,omitempty"`

	// CPU is the container's CPU request (for example "500m"). No CPU limit is set.
	// +optional
	CPU *resource.Quantity `json:"cpu,omitempty"`

	// Memory is the container's memory request and limit (for example "4Gi").
	// +optional
	Memory *resource.Quantity `json:"memory,omitempty"`

	// BackoffLimit is how many times a failed pod is retried. Default 0:
	// a failed GPU job usually fails the same way again.
	// +kubebuilder:default=0
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=10
	// +optional
	BackoffLimit *int32 `json:"backoffLimit,omitempty"`

	// ActiveDeadlineSeconds caps the Job's run time, counted from the Job's start
	// (time spent queued for a GPU counts too).
	// +kubebuilder:validation:Minimum=1
	// +optional
	ActiveDeadlineSeconds *int64 `json:"activeDeadlineSeconds,omitempty"`

	// TTLSecondsAfterFinished deletes the finished Job (and its pods) after this
	// many seconds. The GpuJob itself is kept with its final status.
	// +kubebuilder:validation:Minimum=0
	// +optional
	TTLSecondsAfterFinished *int32 `json:"ttlSecondsAfterFinished,omitempty"`
}

// GpuJobPhase is a one-word summary of where the GpuJob is.
// +kubebuilder:validation:Enum=Pending;Running;Succeeded;Failed
type GpuJobPhase string

const (
	// PhasePending: the Job exists but no pod is running yet (queued for a GPU, pulling the image, ...).
	PhasePending GpuJobPhase = "Pending"
	// PhaseRunning: a pod of the Job is running.
	PhaseRunning GpuJobPhase = "Running"
	// PhaseSucceeded: the Job completed.
	PhaseSucceeded GpuJobPhase = "Succeeded"
	// PhaseFailed: the Job failed (backoff limit, deadline) or could not be created.
	PhaseFailed GpuJobPhase = "Failed"
)

// Condition types set on GpuJob.status.conditions.
const (
	// ConditionJobCreated is True once the owned batch/v1 Job exists.
	ConditionJobCreated = "JobCreated"
	// ConditionScheduled is False while the pod is unschedulable (for example
	// "Insufficient nvidia.com/gpu": the job is queued) and True once it is bound to a node.
	ConditionScheduled = "Scheduled"
	// ConditionFinished is True when the Job has finished; the reason is Succeeded or Failed.
	ConditionFinished = "Finished"
)

// GpuJobStatus is written by the controller only (status subresource).
type GpuJobStatus struct {
	// Phase is Pending, Running, Succeeded or Failed.
	// +optional
	Phase GpuJobPhase `json:"phase,omitempty"`

	// JobName is the name of the owned batch/v1 Job.
	// +optional
	JobName string `json:"jobName,omitempty"`

	// StartTime is when the Job started (copied from the Job).
	// +optional
	StartTime *metav1.Time `json:"startTime,omitempty"`

	// CompletionTime is when the Job finished, successfully or not.
	// +optional
	CompletionTime *metav1.Time `json:"completionTime,omitempty"`

	// Message is a human-readable detail for the current phase.
	// +optional
	Message string `json:"message,omitempty"`

	// ObservedGeneration is the .metadata.generation the status was computed from.
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// Conditions: JobCreated, Scheduled, Finished.
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// GpuJob runs one container on nvidia.com/gpu units as a batch/v1 Job.
//
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=gj
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="GPUs",type=integer,JSONPath=`.spec.gpus`
// +kubebuilder:printcolumn:name="Job",type=string,JSONPath=`.status.jobName`
// +kubebuilder:printcolumn:name="Started",type=date,JSONPath=`.status.startTime`
// +kubebuilder:printcolumn:name="Message",type=string,JSONPath=`.status.message`,priority=1
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`
type GpuJob struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// +kubebuilder:validation:XValidation:rule="self == oldSelf",message="spec is immutable; delete and re-create the GpuJob"
	Spec   GpuJobSpec   `json:"spec"`
	Status GpuJobStatus `json:"status,omitempty"`
}

// GpuJobList is a list of GpuJobs.
//
// +kubebuilder:object:root=true
type GpuJobList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []GpuJob `json:"items"`
}

func init() {
	SchemeBuilder.Register(&GpuJob{}, &GpuJobList{})
}
