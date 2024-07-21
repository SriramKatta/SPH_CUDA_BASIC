import os
import math

def initialize_script_environment():
    """Initialize the script environment by setting the correct working directory."""
    script_directory = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_directory)

def create_particle(x, y, z, mass=0.012, ghost=1):
    """Create a single particle with specified parameters."""
    return {
        'x': round(x, 2),
        'y': round(y, 2),
        'z': round(z, 2),
        'vx': 0.0,
        'vy': 0.0,
        'vz': 0.0,
        'mass': mass,
        'ghost': ghost
    }

def generate_sphere_particles(center, radius, spacing):
    """Generate particles arranged in a sphere without clashing."""
    particles = []
    x_center, y_center, z_center = center

    # Calculate the grid limits
    x_min = x_center - radius
    x_max = x_center + radius
    y_min = y_center - radius
    y_max = y_center + radius
    z_min = z_center - radius
    z_max = z_center + radius

    for x in range(int(x_min / spacing), int(x_max / spacing) + 1):
        for y in range(int(y_min / spacing), int(y_max / spacing) + 1):
            for z in range(int(z_min / spacing), int(z_max / spacing) + 1):
                x_pos = x * spacing
                y_pos = y * spacing
                z_pos = z * spacing
                if math.sqrt((x_pos - x_center)**2 + (y_pos - y_center)**2 + (z_pos - z_center)**2) <= radius:
                    particles.append(create_particle(x_pos, y_pos, z_pos, mass=0.012, ghost=0))
    return particles

def generate_filled_wall(x_range, y_range, z_range, spacing, layer_offset):
    """Generate particles to completely fill a wall with an offset for layers."""
    particles = []
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range

    for x in range(int(x_min / spacing), int(x_max / spacing) + 1):
        for y in range(int(y_min / spacing), int(y_max / spacing) + 1):
            for z in range(int(z_min / spacing), int(z_max / spacing) + 1):
                particles.append(create_particle(x * spacing, y * spacing, z * spacing + layer_offset, ghost=1))

    return particles

def generate_boundary_particles(x_range, y_range, z_range, spacing, num_layers=3):
    """Generate particles for the boundary walls of a cuboid with multiple layers."""
    particles = []
    layer_spacing = spacing / num_layers

    for layer in range(num_layers):
        layer_offset = layer * layer_spacing
        # Generate particles on the six faces of the cuboid
        particles += generate_filled_wall(x_range, y_range, (z_range[0], z_range[0]), spacing, layer_offset)  # Front face
        particles += generate_filled_wall(x_range, y_range, (z_range[1], z_range[1]), spacing, layer_offset)  # Back face
        particles += generate_filled_wall((x_range[0], x_range[0]), y_range, z_range, spacing, layer_offset)  # Left face
        particles += generate_filled_wall((x_range[1], x_range[1]), y_range, z_range, spacing, layer_offset)  # Right face
        particles += generate_filled_wall(x_range, (y_range[0], y_range[0]), z_range, spacing, layer_offset)  # Bottom face
        particles += generate_filled_wall(x_range, (y_range[1], y_range[1]), z_range, spacing, layer_offset)  # Top face

    return particles

def write_vtk_header(file, num_particles):
    """Write the header for the VTK file."""
    header = (
        "# vtk DataFile Version 4.0\n"
        "Particle data\n"
        "ASCII\n"
        "DATASET POLYDATA\n"
        f"POINTS {num_particles} float\n"
    )
    file.write(header)

def write_vtk_points(file, particles):
    """Write particle positions to the VTK file."""
    for particle in particles:
        file.write(f"{particle['x']} {particle['y']} {particle['z']}\n")

def write_vtk_scalar_data(file, particles, attribute):
    """Write scalar data to the VTK file."""
    file.write(f"SCALARS {attribute} double\nLOOKUP_TABLE default\n")
    for particle in particles:
        file.write(f"{particle[attribute]}\n")

def write_vtk_vector_data(file, particles, attribute):
    """Write vector data to the VTK file."""
    file.write(f"VECTORS {attribute} double\n")
    for particle in particles:
        file.write(f"{particle['vx']} {particle['vy']} {particle['vz']}\n")

def write_vtk_file(filename, particles):
    """Write the VTK file with the given particles."""
    num_particles = len(particles)
    with open(filename, 'w') as file:
        write_vtk_header(file, num_particles)
        write_vtk_points(file, particles)
        file.write(f"POINT_DATA {num_particles}\n")
        write_vtk_scalar_data(file, particles, 'mass')
        write_vtk_vector_data(file, particles, 'velocity')
        write_vtk_scalar_data(file, particles, 'ghost')

def main():
    """Main function to generate particles and write them to a VTK file."""
    initialize_script_environment()
    
    # Sphere parameters
    sphere_center = (15, 15, 15)
    sphere_radius = 3.5
    sphere_spacing = 0.4  # Adjust the spacing for particle density
    
    # Boundary cuboid parameters
    x_range = (1, 29)
    y_range = (1, 29)
    z_range = (1, 29)
    boundary_spacing = 0.06  # Adjust spacing for particle density
    
    # Generate fluid particles for the sphere
    fluid_particles = generate_sphere_particles(sphere_center, sphere_radius, sphere_spacing)
    num_fluid_particles = len(fluid_particles)
    print(f"Number of fluid particles: {num_fluid_particles}")
    
    # Generate boundary particles for the cuboid
    boundary_particles = generate_boundary_particles(x_range, y_range, z_range, boundary_spacing, num_layers=1)
    num_boundary_particles = len(boundary_particles)
    
    # Combine all particles
    all_particles = fluid_particles + boundary_particles
    num_total_particles = len(all_particles)
    print(f"Number of boundary particles: {num_boundary_particles}")
    print(f"Total number of particles: {num_total_particles}")
    
    # Write the VTK file with both fluid and boundary particles
    write_vtk_file('./sph_scene_setup.vtk', all_particles)

if __name__ == "__main__":
    main()
