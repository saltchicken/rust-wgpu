use bytemuck::{Pod, Zeroable};
use wgpu;

// This MUST match the amplitude in the shader
pub const WAVE_AMPLITUDE: f32 = 0.1;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 2],
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
                format: wgpu::VertexFormat::Float32x2,
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
    // pub fn get(&self, x: u32, y: u32) -> Option<&T> {
    //     if x >= self.width || y >= self.height {
    //         return None;
    //     }
    //     self.data.get((y * self.width + x) as usize)
    // }
    //
    // pub fn get_mut(&mut self, x: u32, y: u32) -> Option<&mut T> {
    //     if x >= self.width || y >= self.height {
    //         return None;
    //     }
    //     self.data.get_mut((y * self.width + x) as usize)
    // }
    //
    // pub fn width(&self) -> u32 {
    //     self.width
    // }
    //
    // pub fn height(&self) -> u32 {
    //     self.height
    // }

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
        data: vec![Vertex { position: pos }],
        width: 1,
        height: 1,
    }
}

pub fn create_vertex_grid(points_x: u32, points_y: u32) -> Grid<Vertex> {
    if points_x == 0 || points_y == 0 {
        return Grid {
            data: Vec::new(),
            width: 0,
            height: 0,
        };
    }

    let rotation_buffer = 2.0_f32.sqrt();
    let buffer = rotation_buffer + WAVE_AMPLITUDE;
    let grid_range = buffer * 2.0;

    let total_vertices = (points_x * points_y) as usize;
    let mut vertices = Vec::with_capacity(total_vertices);

    let step_x = if points_x > 1 {
        grid_range / (points_x - 1) as f32
    } else {
        0.0
    };
    let step_y = if points_y > 1 {
        grid_range / (points_y - 1) as f32
    } else {
        0.0
    };

    for j in 0..points_y {
        let y = if points_y == 1 {
            0.0
        } else {
            -buffer + (j as f32 * step_y)
        };
        for i in 0..points_x {
            let x = if points_x == 1 {
                0.0
            } else {
                -buffer + (i as f32 * step_x)
            };
            vertices.push(Vertex { position: [x, y] });
        }
    }
    Grid {
        data: vertices,
        width: points_x,
        height: points_y,
    }
}
