from dolfin import *
from mshr import *

# Create mesh
domain = Rectangle(Point(0., 0.), Point(1., 1.))
for i in range(4):
    for j in range(4):
        domain.set_subdomain(i + j*4 + 1, Rectangle(Point(i/4., j/4.), Point((i+1)/4., (j+1)/4.)))
mesh = generate_mesh(domain, 32)
plot(mesh)
interactive()

# Create subdomains
subdomains = MeshFunction("size_t", mesh, 2, mesh.domains())
plot(subdomains)
interactive()

# Create boundaries
class Left(SubDomain):
    def __init__(self, y_min, y_max):
        SubDomain.__init__(self)
        self.y_min = y_min
        self.y_max = y_max
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - 0.) < DOLFIN_EPS and x[1] >= self.y_min and x[1] <= self.y_max

class Right(SubDomain):
    def __init__(self, y_min, y_max):
        SubDomain.__init__(self)
        self.y_min = y_min
        self.y_max = y_max
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - 1.) < DOLFIN_EPS and x[1] >= self.y_min and x[1] <= self.y_max

class Bottom(SubDomain):
    def __init__(self, x_min, x_max):
        SubDomain.__init__(self)
        self.x_min = x_min
        self.x_max = x_max
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - 0.) < DOLFIN_EPS and x[0] >= self.x_min and x[0] <= self.x_max
                
class Top(SubDomain):
    def __init__(self, x_min, x_max):
        SubDomain.__init__(self)
        self.x_min = x_min
        self.x_max = x_max
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - 1.) < DOLFIN_EPS and x[0] >= self.x_min and x[0] <= self.x_max
        
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
for i in range(4):
    left = Left(i/4., (i+1)/4.)
    left.mark(boundaries, i+1)
    top = Top(i/4., (i+1)/4.)
    top.mark(boundaries, i+5)
    right = Right(i/3., (i+1)/3.)
    right.mark(boundaries, 12-i)
    bottom = Bottom(i/3., (i+1)/3.)
    bottom.mark(boundaries, 16-i)
plot(boundaries)
interactive()

# Save
File("16_tblock.xml") << mesh
File("16_tblock_physical_region.xml") << subdomains
File("16_tblock_facet_region.xml") << boundaries
