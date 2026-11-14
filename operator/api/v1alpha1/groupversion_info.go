// Package v1alpha1 holds the GpuJob API (group gpu.zakiscoding.dev, version v1alpha1).
//
// +kubebuilder:object:generate=true
// +groupName=gpu.zakiscoding.dev
package v1alpha1

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/scheme"
)

var (
	// GroupVersion is the API group and version of GpuJob.
	GroupVersion = schema.GroupVersion{Group: "gpu.zakiscoding.dev", Version: "v1alpha1"}

	// SchemeBuilder registers the types of this package with a runtime.Scheme.
	SchemeBuilder = &scheme.Builder{GroupVersion: GroupVersion}

	// AddToScheme adds the types of this package to a runtime.Scheme.
	AddToScheme = SchemeBuilder.AddToScheme
)
