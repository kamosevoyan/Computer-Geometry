import numpy

def norm(x):
    
    if not isinstance(x, numpy.ndarray):
        raise TypeError(f"expected x to be type of numpy.ndarray, but got {type(x)}")
        
    if not ((x.ndim == 2) and (x.shape[-1] == 2)):
        raise ValueError(f"expected x to have shape [N, 2], but got {x.shape}")
 
    return numpy.sqrt(numpy.square(x).sum(axis=-1))

def polygonal_chain(points, density=100):
    
    if not isinstance(points, numpy.ndarray):
        raise TypeError(f"expected points to be numpy.ndarray type, but got {type(points)}")
    
    if not (points.shape[1] == 2) or not (points.ndim == 2):
        raise ValueError(f"expected points shape to be [N, 2], but got {points.shape}")
        
    if not isinstance(density, (int)):
        raise TypeError(f"expected density to be float type, but got {type(density)}")

    if not (0 < density <= 1000):
        raise ValueError(f"expected density to be in range (0, 1000], but got value {density}")

    mask = numpy.array([[i, i + 1] for i in range(points.shape[0] - 1)])
  
    t = numpy.linspace(0, 1, density)
    T = numpy.stack([1 - t, t])  
    
    POINTS = (points[mask].transpose(2, 0, 1) @ T).reshape(2, -1)
    
    return POINTS
        
def ermit_spline(points, tangents, density=100):
    
    if not isinstance(points, numpy.ndarray):
        raise TypeError(f"expected points to be numpy.ndarray type, but got {type(points)}")
        
    if not isinstance(tangents, numpy.ndarray):
        raise TypeError(f"expected tangents to be numpy.ndarray type, but got {type(points)}")
        
    if not (points.shape[1] == 2) or not (points.ndim == 2):
        raise ValueError(f"expected points shape to be [N, 2], but got {points.shape}")
      
    if not (tangents.shape[1] == 2) or not (tangents.ndim == 2):
        raise ValueError(f"expected tangents shape to be [N, 2], but got {tangents.shape}")
        
    if points.shape[0] != tangents.shape[0]:
        raise ValueError(f"expected points and tangents to have same 0 dim value, but got {points.shape[0]} and {tangents.shape[0]}")

    if not isinstance(density, (int)):
        raise TypeError(f"expected density to be float type, but got {type(density)}")

    if not (0 < density <= 1000):
        raise ValueError(f"expected density to be in range (0, 1000], but got value {density}")

    ermits_matrix = numpy.array([
                                    [ 1,  0,  0,  0],
                                    [ 0,  0,  1,  0],
                                    [-3,  3, -2, -1],
                                    [ 2, -2,  1,  1]
                                ])
    
    mask = numpy.array([[i, i + 1] for i in range(points.shape[0] - 1)])

    temp = numpy.stack([points[mask], tangents[mask]], axis=1).reshape(-1, 4, 2)
    
    coefs = ermits_matrix @ temp
    
    t = numpy.linspace(0, 1, density)
    T = numpy.stack([numpy.ones(t.shape[0]), t, t**2, t**3])
    
    POINTS = (coefs.transpose(2, 0, 1) @ T).reshape(2, -1)
    
    return POINTS

def cubic_spline(points, edge_0=None, edge_1=None, edge_type="second", density=100):
    
    if not isinstance(edge_type, str):
        raise TypeError(f"expected edge_type to be type of string, but got {type(edge_type)}")
        
    if edge_type not in ["first", "second"]:
        raise ValueError(f"unknowen edge type {edge_type}")
        
    if not isinstance(points, numpy.ndarray):
        raise TypeError(f"expected points to be numpy.ndarray type, but got {type(points)}")
    
    if not (points.shape[1] == 2) or not (points.ndim == 2):
        raise ValueError(f"expected points shape to be [N, 2], but got {points.shape}")
        
    if (edge_0 is None) ^ (edge_1 is None):
        raise ValueError("both edge_0 and edge_1 should be None or not None simultaneously.")

    if edge_0 is not None:
        if not isinstance(edge_0, (int, float)):
            raise ValueError(f"expected edge_0 to be float or int type, but got {type(edge_0)}")

        if not isinstance(edge_1, (int, float)):
            raise ValueError(f"expected edge_1 to be float or int type, but got {type(edge_1)}")

    if not isinstance(density, (int)):
        raise TypeError(f"expected density to be float type, but got {type(density)}")

    if not (0 < density <= 1000):
        raise ValueError(f"expected density to be in range (0, 1000], but got value {density}")

    N = points.shape[0]

    #the omega parameter
    t = numpy.linspace(0, 1, density)
    T = numpy.stack([numpy.ones(t.shape[0]), t, t**2, t**3])

    #template block
    template = numpy.array([
                                [1,  0,  0,  0,  0,  0,  0,  0],
                                [1,  1,  1,  1,  0,  0,  0,  0],
                                [0,  1,  2,  3,  0, -1,  0,  0],
                                [0,  0,  2,  6,  0,  0, -2,  0]
                           ])

    #mask
    mask = numpy.array([[i, i + 1] for i in range(N - 1)])

    #initialise empty matrix
    main_matrix = numpy.zeros((4 * (N - 1), 4 * (N - 1)))

    #fill values with blocks
    for i in range(N - 2):

        main_matrix[i * 4:(i + 1) * 4, i * 4 : i * 4 + 8] = template

    #fill last interpolation conditions.
    main_matrix[-4:-2, -4:] = template[0:2, 0:4]

    
    if edge_type == "first":
        #edge values for first derivative
        main_matrix[-2][1] = 1
        main_matrix[-1][-4:] = numpy.array([0, 1, 2, 3])
    else:
        #edge values for second derivative
        main_matrix[-2][2] = 2
        main_matrix[-1][-4:] = numpy.array([0, 0, 2, 6])

    #initialize empty vector
    b = numpy.zeros((4 * (N - 1), 2))

    #fill interpolation values
    b[0::4] = points[:-1]
    b[1::4] = points[1:]

    if edge_0 is not None:
        b[-2] = edge_0
        b[-1] = edge_1

    #solve system of linear equations.
    coefs = numpy.linalg.solve(main_matrix, b)
    POINTS = (coefs.reshape(-1, 4, 2).transpose(0, 2, 1) @ T).transpose(1, 0, 2).reshape(2, -1)

    return POINTS

def besier_curve(points, density=100):
    
    ################################################
    def bershtein_basis(n, i, t):
    
        C_n_i = numpy.math.factorial(n)/(numpy.math.factorial(i)*numpy.math.factorial(n - i))
    
        return C_n_i * t ** i * (1 - t) ** (n - i)
    
    ################################################
    
    if not isinstance(points, numpy.ndarray):
        raise TypeError(f"expected points to be numpy.ndarray type, but got {type(points)}")
    
    if not (points.shape[1] == 2) or not (points.ndim == 2):
        raise ValueError(f"expected points shape to be [N, 2], but got {points.shape}")

    if not isinstance(density, (int)):
        raise TypeError(f"expected density to be float type, but got {type(density)}")

    if not (0 < density <= 1000):
        raise ValueError(f"expected density to be in range (0, 1000], but got value {density}")

        
    t = numpy.linspace(0, 1, density)
    result = numpy.zeros((t.shape[0], 2))
    n = points.shape[0]
    
    
    for i, P in enumerate(points):
        
        result += bershtein_basis(n - 1, i, t)[..., None] * P
        
    return result.T