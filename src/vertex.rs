use bytemuck::{Pod, Zeroable};
use wgpu;
// ‼️ This const is no longer needed here, the shader controls the amplitude
// pub const WAVE_AMPLITUDE: f32 = 0.1;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3], // ‼️ Changed from [f32; 2]
}

impl Vertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3, // ‼️ Changed from Float32x2
            }],
        }
    }
}

#[derive(Debug)]
pub struct Grid<T> {
    data: Vec<T>,
    width: u32,
    height: u32,
}

impl<T> Grid<T> {
    // ... (no changes to Grid impl methods) ...
    pub fn as_flat_vec(&self) -> &Vec<T> {
        &self.data
    }
}
// impl Grid<Vertex> {
//     /// Updates a vertex on the CPU grid and returns the modified vertex
//     /// and its flat index for GPU synchronization.
//     pub fn update_vertex(&mut self, x: u32, y: u32, new_pos: [f32; 2]) -> Option<(Vertex, usize)> {
//         if x >= self.width || y >= self.height {
//             return None;
//         }
//         let index = (y * self.width + x) as usize;
//         if let Some(vertex) = self.data.get_mut(index) {
//             vertex.position = new_pos;
//             // Return a *copy* of the modified vertex and its index
//             Some((*vertex, index))
//         } else {
//             None
//         }
//     }
// }

impl<T: Clone> Clone for Grid<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            width: self.width,
            height: self.height,
        }
    }
}
pub fn create_single_point(pos: [f32; 2]) -> Grid<Vertex> {
    Grid {
        // ‼️ Create a 3D point on the X-Z plane
        data: vec![Vertex {
            position: [pos[0], 0.0, pos[1]],
        }],
        width: 1,
        height: 1,
    }
}

pub fn create_vertex_grid(points_x: u32, points_y: u32, step_x: f32, step_y: f32) -> Grid<Vertex> {
    if points_x == 0 || points_y == 0 {
        return Grid {
            data: Vec::new(),
            width: 0,
            height: 0,
        };
    }

    let total_vertices = (points_x * points_y) as usize;
    let mut vertices = Vec::with_capacity(total_vertices);

    for j in 0..points_y {
        let z = j as f32 * step_y + 0.1;
        for i in 0..points_x {
            let x = i as f32 * step_x + 0.1;
            vertices.push(Vertex {
                position: [x, 0.0, z],
            });
        }
    }

    Grid {
        data: vertices,
        width: points_x,
        height: points_y,
    }
}
