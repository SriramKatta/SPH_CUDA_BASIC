import os

def initialize_script_environment():
    """Initialize the script environment by setting the correct working directory."""
    script_directory = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_directory)

def create_particle(x, y, z, mass=500.0, ghost=1):
    """Create a single particle with specified parameters."""
    return {
        'x': round(x, 4),
        'y': round(y, 4),
        'z': round(z, 4),
        'vx': 0.0,
        'vy': 0.0,
        'vz': 0.0,
        'mass': mass,
        'ghost': ghost
    }

def generate_filled_wall(x_range, y_range, z_range, spacing):
    """Generate particles to completely fill a wall."""
    particles = []
    for x in range(x_range[0], x_range[1] + 1, spacing):
        for y in range(y_range[0], y_range[1] + 1, spacing):
            for z in range(z_range[0], z_range[1] + 1, spacing):
                particles.append(create_particle(x, y, z, ghost=1))
    return particles

def generate_boundary_particles(x_range, y_range, z_range, spacing):
    """Generate particles for the boundary walls of a cuboid."""
    particles = []
    
    # Generate particles on the five faces of the cuboid (excluding the top face)
    particles += generate_filled_wall((x_range[0], x_range[1]), (y_range[0], y_range[1]), (z_range[0], z_range[0]), spacing)  # Front face
    particles += generate_filled_wall((x_range[0], x_range[1]), (y_range[0], y_range[1]), (z_range[1], z_range[1]), spacing)  # Back face
    particles += generate_filled_wall((x_range[0], x_range[0]), (y_range[0], y_range[1]), (z_range[0], z_range[1]), spacing)  # Left face
    particles += generate_filled_wall((x_range[1], x_range[1]), (y_range[0], y_range[1]), (z_range[0], z_range[1]), spacing)  # Right face
    particles += generate_filled_wall((x_range[0], x_range[1]), (y_range[0], y_range[0]), (z_range[0], z_range[1]), spacing)  # Bottom face

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
    
    x_range = (1, 29)
    y_range = (1, 29)
    z_range = (1, 29)
    spacing = 1  # Adjust spacing for particle density
    
    # Generate boundary particles only on the outer walls
    boundary_particles = generate_boundary_particles(x_range, y_range, z_range, spacing)
    
    # Write the VTK file with only boundary particles
    write_vtk_file('./boundary_particles.vtk', boundary_particles)

if __name__ == "__main__":
    main()
