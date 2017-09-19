from dolfin import *
from mshr import *

# Create mesh
domain = Rectangle(Point(0., 0.), Point(1., 1.))
domain.set_subdomain(1, Rectangle(Point(0., 0.), Point(1., 1.)))
mesh = generate_mesh(domain, 50)
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
left = Left(0., 1.)
left.mark(boundaries, 1)
top = Top(0., 1.)
top.mark(boundaries, 2)
right = Right(0., 1.)
right.mark(boundaries, 3)
bottom = Bottom(0., 1.)
bottom.mark(boundaries, 4)
plot(boundaries)
interactive()

# Save
File("1_tblock.xml") << mesh
File("1_tblock_physical_region.xml") << subdomains
File("1_tblock_facet_region.xml") << boundaries
