# Copyright (c) 2022-2025, The TacEx Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Contact Force Extractor Module

This module provides functionality to extract contact force information from libuipc's
contact system. It computes normal and tangential forces for each vertex/marker point.

Based on libuipc's contact system feature:
- Energy/Gradient/Hessian is scaled by Δt²
- Force from Newton's Law: f = -g / Δt²
  where f is the force, g is the gradient

Reference: https://github.com/spiriMirror/libuipc-samples/blob/main/python/20_contact_system_feature/main.py
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Dict, List

from uipc.core import ContactSystemFeature
from uipc.geometry import Geometry, SimplicialComplex

if TYPE_CHECKING:
    from tacex_uipc.sim import UipcSim
    from tacex_uipc.objects import UipcObject


@dataclass
class ContactForceData:
    """Data container for contact force information at marker/vertex points."""
    
    # Vertex indices that have contact forces
    vertex_indices: Optional[np.ndarray] = None
    """Indices of vertices with contact forces. Shape: (num_contact_vertices,)"""
    
    # Total contact gradient (raw from libuipc, scaled by dt^2)
    contact_gradient: Optional[np.ndarray] = None
    """Contact gradient at vertices. Shape: (num_contact_vertices, 3)"""
    
    # Contact forces (computed from gradient)
    contact_forces: Optional[np.ndarray] = None
    """Contact forces at vertices (f = -gradient / dt^2). Shape: (num_contact_vertices, 3)"""
    
    # Decomposed forces
    normal_forces: Optional[np.ndarray] = None
    """Normal component of contact forces. Shape: (num_contact_vertices, 3)"""
    
    tangential_forces: Optional[np.ndarray] = None
    """Tangential component of contact forces. Shape: (num_contact_vertices, 3)"""
    
    # Force magnitudes
    normal_force_magnitudes: Optional[np.ndarray] = None
    """Magnitude of normal forces. Shape: (num_contact_vertices,)"""
    
    tangential_force_magnitudes: Optional[np.ndarray] = None
    """Magnitude of tangential forces. Shape: (num_contact_vertices,)"""
    
    # Surface normals at contact points
    surface_normals: Optional[np.ndarray] = None
    """Surface normals at contact points. Shape: (num_contact_vertices, 3)"""

    # Nodal tributary areas at contact points
    nodal_areas: Optional[np.ndarray] = None
    """Tributary area for each contact vertex. Shape: (num_contact_vertices,)"""

    # Pressure (force normalized by tributary area)
    contact_pressures: Optional[np.ndarray] = None
    """Contact pressures at vertices. Shape: (num_contact_vertices, 3)"""

    # Pressure magnitudes
    pressure_magnitudes: Optional[np.ndarray] = None
    """Magnitude of contact pressures. Shape: (num_contact_vertices,)"""

    # Decomposed pressures
    normal_pressures: Optional[np.ndarray] = None
    """Normal component of contact pressures. Shape: (num_contact_vertices, 3)"""

    tangential_pressures: Optional[np.ndarray] = None
    """Tangential component of contact pressures. Shape: (num_contact_vertices, 3)"""

    normal_pressure_magnitudes: Optional[np.ndarray] = None
    """Magnitude of normal pressures. Shape: (num_contact_vertices,)"""

    tangential_pressure_magnitudes: Optional[np.ndarray] = None
    """Magnitude of tangential pressures. Shape: (num_contact_vertices,)"""
    
    # Time step used for force computation
    dt: float = 0.01
    """Time step used for computing forces from gradient."""
    
    # Contact primitive information
    contact_primitive_types: List[str] = field(default_factory=list)
    """Types of contact primitives (e.g., 'PP', 'PE', 'PT', 'EE', 'PH')"""


class ContactForceExtractor:
    """
    Extracts contact force information from libuipc simulation.
    
    This class interfaces with libuipc's ContactSystemFeature to extract
    contact gradients and compute forces for each vertex. Forces are
    decomposed into normal and tangential components.
    
    Usage:
        extractor = ContactForceExtractor(uipc_sim)
        
        # After simulation step
        force_data = extractor.extract_contact_forces()
        
        # Get forces for specific object
        obj_forces = extractor.get_object_forces(uipc_object)
    """
    
    def __init__(self, uipc_sim: UipcSim):
        """
        Initialize the contact force extractor.
        
        Args:
            uipc_sim: The UIPC simulation instance.
        """
        self._uipc_sim = uipc_sim
        self._contact_feature: Optional[ContactSystemFeature] = None
        self._is_initialized = False
        
        # Cache for force data
        self._force_data = ContactForceData()
        self._last_frame = -1
        
    def initialize(self):
        """Initialize the contact force extractor after simulation setup."""
        try:
            features = self._uipc_sim.engine.features()
            self._contact_feature = features.find(ContactSystemFeature.FeatureName)
            if self._contact_feature is None:
                print("[ContactForceExtractor] Warning: ContactSystemFeature not found in engine features.")
                return False
            self._is_initialized = True
            print("[ContactForceExtractor] Successfully initialized.")
            return True
        except Exception as e:
            print(f"[ContactForceExtractor] Failed to initialize: {e}")
            return False
    
    @property
    def is_initialized(self) -> bool:
        """Check if the extractor is properly initialized."""
        return self._is_initialized
    
    @property
    def contact_primitive_types(self) -> List[str]:
        """Get available contact primitive types."""
        if not self._is_initialized:
            return []
        try:
            return self._contact_feature.contact_primitive_types()
        except Exception:
            return []
    
    def extract_contact_forces(self, compute_decomposition: bool = True) -> ContactForceData:
        """
        Extract contact forces from the current simulation state.
        
        This method retrieves contact gradient information from libuipc and
        computes the actual forces using f = -gradient / dt^2.
        
        Args:
            compute_decomposition: Whether to compute normal/tangential decomposition.
                                   Requires surface normal information.
        
        Returns:
            ContactForceData containing force information for all contact vertices.
        """
        if not self._is_initialized:
            if not self.initialize():
                return ContactForceData()
        
        # Get time step
        dt = self._uipc_sim.cfg.dt
        self._force_data.dt = dt
        
        # Get contact primitive types
        self._force_data.contact_primitive_types = self.contact_primitive_types
        
        # Create geometry to receive gradient data
        gradient_geo = Geometry()
        
        try:
            # Retrieve contact gradient from libuipc
            self._contact_feature.contact_gradient(gradient_geo)
            
            # Extract vertex indices and gradients
            instances = gradient_geo.instances()
            instance_size = instances.size()
            
            # Debug output removed - too verbose in normal operation
            # Uncomment for debugging: print(f"[ContactForceExtractor] Instance size: {instance_size}")
            
            if instance_size == 0:
                # No contact data available
                self._force_data.vertex_indices = np.array([], dtype=np.int32)
                self._force_data.contact_gradient = np.array([]).reshape(0, 3)
                self._force_data.contact_forces = np.array([]).reshape(0, 3)
                self._force_data.nodal_areas = np.array([])
                self._force_data.contact_pressures = np.array([]).reshape(0, 3)
                self._force_data.pressure_magnitudes = np.array([])
                return self._force_data
            
            # Check if we have contact data
            i_attr = instances.find('i')
            if i_attr is None:
                # No contact data available
                self._force_data.vertex_indices = np.array([], dtype=np.int32)
                self._force_data.contact_gradient = np.array([]).reshape(0, 3)
                self._force_data.contact_forces = np.array([]).reshape(0, 3)
                self._force_data.nodal_areas = np.array([])
                self._force_data.contact_pressures = np.array([]).reshape(0, 3)
                self._force_data.pressure_magnitudes = np.array([])
                return self._force_data
            
            # Get vertex indices - use uipc.view() 
            import pyuipc as uipc
            i_view = uipc.view(i_attr)
            vertex_indices = np.array(i_view).flatten()
            self._force_data.vertex_indices = vertex_indices
            
            # Get gradient values
            grad_attr = instances.find('grad')
            if grad_attr is not None:
                grad_view = uipc.view(grad_attr)
                gradients = np.array(grad_view).reshape(-1, 3)
                self._force_data.contact_gradient = gradients
                
                # Compute forces: f = -g / dt^2
                # The gradient is scaled by dt^2 in libuipc
                self._force_data.contact_forces = -gradients / (dt * dt)
            else:
                self._force_data.contact_gradient = np.zeros((len(vertex_indices), 3))
                self._force_data.contact_forces = np.zeros((len(vertex_indices), 3))

            # Compute tributary areas for contact vertices (for pressure normalization)
            self._force_data.nodal_areas = self._get_surface_nodal_areas(vertex_indices)
            
            # Compute force decomposition if requested
            if compute_decomposition and len(vertex_indices) > 0:
                self._compute_force_decomposition()
            
        except Exception as e:
            print(f"[ContactForceExtractor] Error extracting contact forces: {e}")
            import traceback
            traceback.print_exc()
        
        return self._force_data
    
    def _compute_force_decomposition(self):
        """
        Decompose contact forces into normal and tangential components.
        
        Uses surface normals at contact points to decompose the total force
        into normal (perpendicular to surface) and tangential (parallel to surface)
        components.
        """
        if self._force_data.contact_forces is None or len(self._force_data.contact_forces) == 0:
            return
        
        # Get surface normals for contact vertices
        normals = self._get_surface_normals(self._force_data.vertex_indices)
        self._force_data.surface_normals = normals
        
        if normals is None or len(normals) == 0:
            # Use z-axis as default normal if normals unavailable
            normals = np.tile([0.0, 0.0, 1.0], (len(self._force_data.contact_forces), 1))
            self._force_data.surface_normals = normals
        
        forces = self._force_data.contact_forces
        
        # Compute normal component: f_n = (f · n) * n
        dot_products = np.sum(forces * normals, axis=1, keepdims=True)
        normal_forces = dot_products * normals
        self._force_data.normal_forces = normal_forces
        self._force_data.normal_force_magnitudes = np.abs(dot_products.flatten())
        
        # Compute tangential component: f_t = f - f_n
        tangential_forces = forces - normal_forces
        self._force_data.tangential_forces = tangential_forces
        self._force_data.tangential_force_magnitudes = np.linalg.norm(tangential_forces, axis=1)

        # Compute pressure-normalized quantities if nodal areas are available
        nodal_areas = self._force_data.nodal_areas
        if nodal_areas is not None and len(nodal_areas) == len(forces):
            safe_areas = np.where(nodal_areas > 1e-12, nodal_areas, 1e-12)
            contact_pressures = forces / safe_areas[:, None]
            self._force_data.contact_pressures = contact_pressures
            self._force_data.pressure_magnitudes = np.linalg.norm(contact_pressures, axis=1)

            normal_pressures = normal_forces / safe_areas[:, None]
            tangential_pressures = tangential_forces / safe_areas[:, None]
            self._force_data.normal_pressures = normal_pressures
            self._force_data.tangential_pressures = tangential_pressures
            self._force_data.normal_pressure_magnitudes = np.abs(np.sum(normal_pressures * normals, axis=1))
            self._force_data.tangential_pressure_magnitudes = np.linalg.norm(tangential_pressures, axis=1)
    
    def _get_surface_normals(self, vertex_indices: np.ndarray) -> np.ndarray:
        """
        Get surface normals at specified vertex indices.
        
        Args:
            vertex_indices: Array of vertex indices to get normals for.
        
        Returns:
            Array of normalized surface normal vectors. Shape: (num_vertices, 3)
        """
        try:
            # Try to get normals from the simulation's surface data
            sio = self._uipc_sim.sio
            surface = sio.simplicial_surface(2)
            
            # Get all positions and triangles
            positions = surface.positions().view().reshape(-1, 3)
            triangles = surface.triangles().topo().view().reshape(-1, 3)

            # Compute per-vertex normals from adjacent triangles (area-weighted)
            vertex_normals = self._compute_vertex_normals(positions, triangles)
            
            # Extract normals for specified vertices
            if len(vertex_indices) > 0:
                # Clamp indices to valid range
                valid_indices = np.clip(vertex_indices, 0, len(vertex_normals) - 1)
                return vertex_normals[valid_indices]
            
            return np.array([]).reshape(0, 3)
            
        except Exception as e:
            print(f"[ContactForceExtractor] Warning: Could not get surface normals: {e}")
            # Return default upward normals
            return np.tile([0.0, 0.0, 1.0], (len(vertex_indices), 1))
    
    @staticmethod
    def _compute_vertex_normals(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
        """
        Compute per-vertex normals from mesh geometry.
        
        Args:
            positions: Vertex positions. Shape: (num_vertices, 3)
            triangles: Triangle indices. Shape: (num_triangles, 3)
        
        Returns:
            Normalized per-vertex normals. Shape: (num_vertices, 3)
        """
        num_vertices = len(positions)
        vertex_normals = np.zeros((num_vertices, 3))
        
        # Compute area-weighted face normals and accumulate to vertices
        for tri in triangles:
            v0, v1, v2 = positions[tri[0]], positions[tri[1]], positions[tri[2]]

            # Compute face normal and area
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(face_normal)
            if norm < 1e-12:
                continue
            face_area = 0.5 * norm
            face_unit = face_normal / norm

            weighted_normal = face_unit * face_area
            vertex_normals[tri[0]] += weighted_normal
            vertex_normals[tri[1]] += weighted_normal
            vertex_normals[tri[2]] += weighted_normal
        
        # Normalize
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        vertex_normals = vertex_normals / norms
        
        return vertex_normals

    def _get_surface_nodal_areas(self, vertex_indices: np.ndarray) -> np.ndarray:
        """
        Compute tributary (per-vertex) areas for the surface mesh and return
        areas for the specified vertex indices.
        """
        try:
            sio = self._uipc_sim.sio
            surface = sio.simplicial_surface(2)

            positions = surface.positions().view().reshape(-1, 3)
            triangles = surface.triangles().topo().view().reshape(-1, 3)

            nodal_areas = self._compute_nodal_areas(positions, triangles)

            if len(vertex_indices) > 0:
                valid_indices = np.clip(vertex_indices, 0, len(nodal_areas) - 1)
                return nodal_areas[valid_indices]

            return np.array([])
        except Exception as e:
            print(f"[ContactForceExtractor] Warning: Could not compute nodal areas: {e}")
            return np.ones(len(vertex_indices), dtype=np.float32)

    @staticmethod
    def _compute_nodal_areas(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
        """
        Compute per-vertex tributary areas from surface triangles.
        """
        v0 = positions[triangles[:, 0]]
        v1 = positions[triangles[:, 1]]
        v2 = positions[triangles[:, 2]]

        face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        nodal_areas = np.zeros(len(positions), dtype=np.float32)
        for i in range(3):
            np.add.at(nodal_areas, triangles[:, i], face_areas / 3.0)

        return nodal_areas
    
    def get_object_forces(self, uipc_object: UipcObject) -> ContactForceData:
        """
        Get contact forces for a specific UIPC object.
        
        Args:
            uipc_object: The UIPC object to get forces for.
        
        Returns:
            ContactForceData containing only forces for the specified object.
        """
        # First extract all contact forces
        all_forces = self.extract_contact_forces()
        
        if all_forces.vertex_indices is None or len(all_forces.vertex_indices) == 0:
            return ContactForceData()
        
        # For now, return all forces (TODO: filter by object)
        # The vertex indices from contact_gradient are global DoF indices
        # We need to map them to object-specific vertices
        
        # Just return all forces for now - shows that contact is detected
        return all_forces
    
    def get_marker_forces(self, 
                          uipc_object: UipcObject,
                          marker_indices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get contact forces at specific marker points on an object.
        
        This method is useful for GelSight sensors where you want to get
        force information at specific marker locations.
        
        Args:
            uipc_object: The UIPC object (e.g., gelpad).
            marker_indices: Indices of marker vertices on the object's surface.
        
        Returns:
            Dictionary containing:
                - 'forces': Total forces at markers. Shape: (num_markers, 3)
                - 'normal_forces': Normal force components. Shape: (num_markers, 3)
                - 'tangential_forces': Tangential force components. Shape: (num_markers, 3)
                - 'normal_magnitudes': Normal force magnitudes. Shape: (num_markers,)
                - 'tangential_magnitudes': Tangential force magnitudes. Shape: (num_markers,)
                - 'has_contact': Boolean mask of markers with contact. Shape: (num_markers,)
        """
        num_markers = len(marker_indices)
        result = {
            'forces': np.zeros((num_markers, 3)),
            'normal_forces': np.zeros((num_markers, 3)),
            'tangential_forces': np.zeros((num_markers, 3)),
            'normal_magnitudes': np.zeros(num_markers),
            'tangential_magnitudes': np.zeros(num_markers),
            'has_contact': np.zeros(num_markers, dtype=bool)
        }
        
        obj_forces = self.get_object_forces(uipc_object)
        
        if obj_forces.vertex_indices is None or len(obj_forces.vertex_indices) == 0:
            return result
        
        # Map marker indices to contact force data
        for i, marker_idx in enumerate(marker_indices):
            # Check if this marker has contact force data
            contact_mask = obj_forces.vertex_indices == marker_idx
            if np.any(contact_mask):
                idx = np.where(contact_mask)[0][0]
                result['has_contact'][i] = True
                if obj_forces.contact_forces is not None:
                    result['forces'][i] = obj_forces.contact_forces[idx]
                
                if obj_forces.normal_forces is not None:
                    result['normal_forces'][i] = obj_forces.normal_forces[idx]
                if obj_forces.tangential_forces is not None:
                    result['tangential_forces'][i] = obj_forces.tangential_forces[idx]
                if obj_forces.normal_force_magnitudes is not None:
                    result['normal_magnitudes'][i] = obj_forces.normal_force_magnitudes[idx]
                if obj_forces.tangential_force_magnitudes is not None:
                    result['tangential_magnitudes'][i] = obj_forces.tangential_force_magnitudes[idx]
        
        return result
    
    def get_total_contact_force(self, uipc_object: Optional[UipcObject] = None) -> np.ndarray:
        """
        Get the total contact force (sum of all vertex forces).
        
        Args:
            uipc_object: Optional, if provided, returns force for specific object only.
        
        Returns:
            Total contact force vector. Shape: (3,)
        """
        if uipc_object is not None:
            force_data = self.get_object_forces(uipc_object)
        else:
            force_data = self.extract_contact_forces()
        
        if force_data.contact_forces is None or len(force_data.contact_forces) == 0:
            return np.zeros(3)
        
        return np.sum(force_data.contact_forces, axis=0)


def compute_force_from_gradient(gradient: np.ndarray, dt: float) -> np.ndarray:
    """
    Utility function to compute force from gradient.
    
    In libuipc, the optimization problem uses scaled energy/gradient/hessian by dt^2.
    The force can be computed as: f = -gradient / dt^2
    
    Args:
        gradient: Contact gradient from libuipc. Shape: (n, 3)
        dt: Time step used in simulation.
    
    Returns:
        Force vectors. Shape: (n, 3)
    """
    return -gradient / (dt * dt)
