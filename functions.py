import numpy
import matplotlib.pyplot as plt

def sigma(m, r, t=0):

    zeros = numpy.zeros(r.shape[0])

    return numpy.maximum(r - t, zeros)**m

def sigma_diff(nodes, x):

    result = numpy.zeros(x.shape[0])
    indices = numpy.arange(nodes.shape[0])

    for j, _ in enumerate(nodes):
        denom = (nodes[j] - nodes[indices != j]).prod()
        result += sigma(nodes.shape[0] - 2, nodes[j] - x)/denom

    return result

def norm_bspline(i, m, T, x):

    return (T[i+m] - T[i]) * sigma_diff(T[i:(i + m + 1)], x)

def construct_curve(new_points, new_weights, nodes, degree, density):

    num = nodes.shape[0] - 2 * degree
    x = numpy.linspace(nodes[degree], nodes[-degree], num=density * num)

    result = numpy.zeros((2, x.shape[0]))

    for index, _ in enumerate(new_points):
        result += new_weights[index] * new_points[index][..., None] * norm_bspline(index, degree, nodes, x)

    return result, num

def nurbs_curve(points, degree, nodes=None, weights=None, density=100, split=True, cascade=False):

    if not isinstance(degree, (int, list)):
        raise TypeError(f"Expected degree to be int or list of ints, but got {type(degree)}")

    if isinstance(degree, int):
        degree = [degree]

    if (len(degree) > 1) and (nodes is not None):
        raise NotImplementedError(f"Multiple degrees with non default nodes are not available")

    if not all(isinstance(num, int) and (2 <= num) for num in degree):
        raise ValueError(f"Expected degree to be list of integers greater or equal than 2")

    if not isinstance(split, bool):
        raise TypeError(f"Expected split to be bool, but got {type(split)}")

    if not isinstance(cascade, bool):
        raise TypeError(f"Expected cascade to be bool, but got {type(cascade)}")

    if not isinstance(density, int):
        raise TypeError(f"Expected density to be int, but got {type(density)}")
    if not (5 <= density <= 1000):
        raise ValueError(f"Expected density to be in range [20, 1000], but got {density}")

    if not isinstance(points, numpy.ndarray):
        raise TypeError(f"Expected points to be numpy array, but got {type(points)}")

    if not ((points.ndim == 2) and (points.shape[1] == 2)):
        raise ValueError(f"Expected points shape to be [N, 2], but got {points.shape}")

    if weights is None:
        weights = numpy.ones(points.shape[0])
    else:
        if not isinstance(weights, numpy.ndarray):
            raise TypeError(f"Expected weights to be numpy array, but got {type(weights)}")
        if weights.shape[0] != points.shape[0]:
            raise ValueError(f"Expectes points and weights to have sampe last shape, but got {points.shape} and {weights.shape}")


    plt.figure(figsize=(10, 10))
    plt.axis("equal")

    for deg in degree:
        #Pad points insead of multiple nodes to escape zero division
        new_points  = numpy.concatenate([[points[0]]*deg, points[1:-1], [points[-1]]*deg])
        new_weights = numpy.concatenate([[weights[0]]*deg, weights[1:-1], [weights[-1]]*deg])

        if nodes is None:
            new_nodes = numpy.arange(new_points.shape[0]+deg)
        else:
            if not isinstance(nodes, numpy.ndarray):
                raise TypeError(f"Expected nodes to be numpy array, but got {type(nodes)}")
            if nodes.ndim != 1:
                raise ValueError(f"Expected nodes shape to be [N], but got {nodes.shape}")
            if not numpy.all(nodes[1:] > nodes[:-1]):
                raise ValueError("Expected nodes to be non strict increasing sequence")

            left_count = deg // 2
            right_count = deg // 2 + (deg % 2)

            left_pad_nodes = numpy.linspace(nodes[0]-1, nodes[0], deg+left_count)
            right_pad_nodes = numpy.linspace(nodes[-1], nodes[-1]+1, deg+right_count)

            new_nodes = numpy.concatenate([left_pad_nodes, nodes[1:-1], right_pad_nodes])

            if new_nodes.shape[0] != new_points.shape[0] + deg:
                raise ValueError(f"Expected node.shape = points.shape + degree")

        result, num = construct_curve(new_points, new_weights, new_nodes, deg, density)
        #Define figure to plot to

        if split is True:

            X = numpy.split(result[0], num)
            Y = numpy.split(result[1], num)

            for x, y in zip(X, Y):
                plt.plot(x, y, linewidth=3)

            if cascade is True:
                plt.plot(points[:, 0], points[:, 1], color="k", linewidth=0.5)

        else:
            plt.plot(result[0], result[1], label=f"{deg}", linewidth=3)
            plt.legend()

        plt.scatter(points[:, 0], points[:, 1], c="r")

    plt.show()

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

def besier_curve(points, weights=None, density=100):
    
    ################################################
    def bershtein_basis(n, i, t):
    
        C_n_i = numpy.math.factorial(n)/(numpy.math.factorial(i) * numpy.math.factorial(n - i))
    
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
        
    if weights is not None:
        if not isinstance(weights, numpy.ndarray):
            raise TypeError(f"expected weights to be numpy.ndarray type, but got {type(points)}")
    
        if not (weights.ndim == 1):
            raise ValueError(f"expected weights to have shape [N], but got {points.shape}")
    else:
        weights = numpy.ones(points.shape[0], dtype=float)

        
    t = numpy.linspace(0, 1, density)
    result = numpy.zeros((t.shape[0], 2))
    n = points.shape[0]
    
    
    for i, P in enumerate(points):
        
        result += weights[i] * bershtein_basis(n - 1, i, t)[..., None] * P
        
    if weights is not None:
        
        denom = numpy.zeros(t.shape[0], dtype=float)
        
        for i, P in enumerate(weights):
            
            denom += weights[i] * bershtein_basis(n - 1, i, t)
         
        result = result /  denom[..., None]
        
    return result.T

def add_one_point(old_points):
    
    if not isinstance(old_points, numpy.ndarray):
        raise TypeError(f"expected old_points to be numpy.ndarray type, but got {type(points)}")
    
    if not (old_points.shape[1] == 2) or not (old_points.ndim == 2):
        raise ValueError(f"expected old_points shape to be [N, 2], but got {points.shape}")

    n = old_points.shape[0]
    new_points = numpy.zeros((n + 1, 2), dtype=float)
    
    new_points[0] = old_points[0]
    
    for i, value in enumerate(old_points[1:], 1):
        
        new_points[i] = (n - i) / (n) * old_points[i] + i / (n) * old_points[i - 1]
        
    new_points[-1] = old_points[-1]
    
    return new_points
