import numpy
import matplotlib.pyplot as plt
from tqdm import tqdm

#This is set for endge cases when 0 / 0 occures. In future, need to be solved.
numpy.seterr(invalid='ignore')

def get_intersections(points, i, j, L):

    b =  (points[i][1]-points[j][1])/(points[j][0]-L)
    a = (1/(points[j][0]-L)-1/(points[i][0]-L))*0.5
    c = (points[i][1]-points[j][1])**2/(points[j][0]-L)*0.5 + (points[j][0]-points[i][0])/2
    D = b**2 - 4*a*c

    if numpy.isclose(a, 0):
        t = -c/b
        y = t + points[i][1]
        x = (y - points[i][1])**2*0.5/(points[i][0]-L)+0.5*(L+points[i][0])

        return ((x, y), )

    t1 = 0.5/a*(-b - numpy.sqrt(D))
    t2 = 0.5/a*(-b + numpy.sqrt(D))

    y1 = t1 + points[i][1]
    y2 = t2 + points[i][1]

    x1 = (y1 - points[i][1])**2*0.5/(points[i][0]-L)+0.5*(L+points[i][0])
    x2 = (y2 - points[i][1])**2*0.5/(points[i][0]-L)+0.5*(L+points[i][0])

    return (x1, y1), (x2, y2)

def voronoi_diagram(points, min_y=-2, max_y=2, epsilon=1e-3):

    if not isinstance(points, numpy.ndarray):
        raise TypeError(f"Expected points to be numpy array, not got {type(points)}")

    if not ((points.ndim == 2) and (points.shape[1] == 2)):
        raise ValueError(f"Expetced points to have shape [N, 2], but got {points.shape}")

    y = numpy.linspace(min_y, max_y, 100_000)

    diagram_points = []

    #rescale points for less computations
    scale = numpy.abs(points).max()
    points = points / scale

    for L in tqdm(numpy.linspace(points[:,0].min(), 10,  10000)):

        crop_indices = []
        POINTS = []
        X = []

        #get those points that are on the right side of the sliding line
        available_indices = numpy.where(L > points[:,0])[0]

        #if there are not points on the right side of the line, skip the iteration
        if available_indices.size <= 1:
            continue

        #get all couples of parabols that intersect
        couples =  numpy.array([[i, j] for j in range(available_indices.shape[0]) for i in range(j)])

        #iterate over all couples
        for first, second in available_indices[couples]:

            #get coordinates of intersection of given two parabols
            result = get_intersections(points, first, second, L)

            #if the points are inside the given screen range
            if (min_y <= result[0][1] <= max_y) and (min_y <= result[0][0] <= max_y):
                POINTS.append(result[0])
                #get those indices whose values are close to the y coordinate of given intersection point
                crop_indices.append(numpy.where(numpy.abs(y - result[0][1]) <= epsilon)[0])

            #if the result is one point, continue
            if len(result) == 1:
                continue

            #the same as in the previous case
            if (min_y <= result[1][1] <= max_y) and (min_y <= result[1][0] <= max_y):
                POINTS.append(result[1])
                crop_indices.append(numpy.where(numpy.abs(y - result[1][1]) <= epsilon)[0])

        #if there are not intersection points, continue
        if len(POINTS) == 0:
            continue

        #for each available point, get x values of parabola
        for index in available_indices:
            x = (y - points[index][1])**2/(2*(points[index][0]-L)) + (L+points[index][0])/2
            X.append(x)

        crop_indices = numpy.array(crop_indices)

        if crop_indices.dtype == object:
            continue

        POINTS = numpy.array(POINTS)
        X = numpy.array(X)

        #these try-catch statements are made because of numeric error, needs to be solved with more precise approach
        try:
            available_plot_indices = ~numpy.any((POINTS[:, 0][None, :, None] > X[:, crop_indices]).sum(axis=-1) == 0, axis=0)
        except Exception as e:
            continue

        if available_plot_indices.size == 0:
            continue

        try:
            resulting_points = POINTS[available_plot_indices]
        except:
            pass

        diagram_points.extend(resulting_points)
    
    
    couples =  numpy.array([[i, j] for j in range(points.shape[0]) for i in range(j)])
    neighbours = [[] for i in range(points.shape[0])]

    for (first, second) in couple_indices:

        alpha = -numpy.arccos((points[second, 0] - points[first, 0])/ \
        numpy.sqrt((points[first, 0] - points[second, 0])**2 + (points[first, 1] - points[second, 1])**2))

        if points[first, 1] < points[second, 1]:
            alpha *= -1

        rot = numpy.array([
        [numpy.cos(alpha), -numpy.sin(alpha)],
        [numpy.sin(alpha), numpy.cos(alpha)]
        ])

        result_rot = result - points[first]
        result_rot = result_rot @ rot
        result_rot += points[second]

        points_rot = points - points[first]
        points_rot = points_rot @ rot
        points_rot += points[second]

        middle_x = 0.5 * (points_rot[first, 0] + points_rot[second, 0])

        a = (numpy.abs(result_rot[numpy.abs(result_rot[:, 1] - points_rot[second, 1]) <= eps][:, 0] - middle_x) <= eps).sum()
        b = (numpy.abs(result_rot[numpy.abs(result_rot[:, 0] - middle_x) <= eps][:, 1] - points_rot[first, 1]) <= eps).sum() 

        tmp = result_rot[numpy.abs(result_rot[:, 1] - points_rot[second, 1]) <= eps]
        jmp = tmp[numpy.where(numpy.abs(result_rot[numpy.abs(result_rot[:, 1] - points_rot[second, 1]) <= eps][:, 0] - middle_x) > 10*eps)]

        min_ = min(points_rot[first, 0], points_rot[second, 0])
        max_ = max(points_rot[first, 0], points_rot[second, 0])

        jmp = jmp[jmp[:, 0] > min_]
        jmp = jmp[jmp[:, 0] < max_]

        if a and (jmp.shape[0] == 0):
            neighbours[first].append(second)
    
    return numpy.stack(diagram_points) * scale, neighbours

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

def rec_bspline(i, m, nodes, x):

    if m == 1:
        result = numpy.zeros(x.shape[0])
        result[(x >= nodes[i]) & (x <= nodes[(i+1)])] = 1
        return result

    beta = i + m

    if nodes[beta] != nodes[i+1]:
        first = (nodes[beta]-x)/(nodes[beta]-nodes[i+1])*rec_bspline(i+1,m-1,nodes,x)
    else:
        first = numpy.zeros(x.shape[0])

    if nodes[beta-1] != nodes[i]:
        second = (x-nodes[i])/(nodes[beta-1]-nodes[i])*rec_bspline(i,m-1,nodes,x)
    else:
        second = numpy.zeros(x.shape[0])

    return first + second

def construct_curve(points, weights, nodes, degree, density):

    num = numpy.unique(nodes).shape[0]
    x = numpy.linspace(nodes[0], nodes[-1], num=density*num)

    nom = numpy.zeros((2, x.shape[0]))
    denom = numpy.zeros(x.shape[0])

    for index, _ in enumerate(points):
        temp = weights[index] * rec_bspline(index, degree, nodes, x)
        nom += points[index][..., None] * temp
        denom += temp

    return nom/denom, num

def nurbs_curve(points, degree, nodes=None, weights=None, density=100, split=True, cascade=False):

    if not isinstance(degree, int):
        raise TypeError(f"Expected degree to be int, but got {type(degree)}")

    if not isinstance(split, bool):
        raise TypeError(f"Expected split to be bool, but got {type(split)}")

    if not isinstance(cascade, bool):
        raise TypeError(f"Expected cascade to be bool, but got {type(cascade)}")

    if not isinstance(density, int):
        raise TypeError(f"Expected density to be int, but got {type(density)}")
    if not (5 <= density <= 1000):
        raise ValueError(f"Expected density to be in range [5, 1000], but got {density}")

    if not isinstance(points, numpy.ndarray):
        raise TypeError(f"Expected points to be numpy array, but got {type(points)}")

    if not ((points.ndim == 2) and (points.shape[1] == 2)):
        raise ValueError(f"Expected points shape to be [N, 2], but got {points.shape}")

    if nodes is None:
        nodes = numpy.arange(points.shape[0] + degree)
    else:
        if not isinstance(nodes, numpy.ndarray):
            raise TypeError(f"Expected nodes to be numpy array, but got {type(nodes)}")

        if nodes.ndim != 1:
            raise ValueError(f"Expected nodes shape to be [N], but got {nodes.shape}")

        if not numpy.all(nodes[1:] >= nodes[:-1]):
            raise ValueError("Expected nodes to be non strictly increasing sequence")

    if  not (2 <= degree <= points.shape[0]):
        raise ValueError(f"The degree of the spline must be in range [2, points.shape[0]]")

    if nodes.shape[0] != points.shape[0] + degree:
        raise ValueError(f"Expected node.shape = points.shape + degree")

    if weights is None:
        weights = numpy.ones(points.shape[0])
    else:
        if not isinstance(weights, numpy.ndarray):
            raise TypeError(f"Expected weights to be numpy array, but got {type(weights)}")
        if weights.shape[0] != points.shape[0]:
            raise ValueError(f"Expectes points and weights to have sampe last shape, but got {points.shape} and {weights.shape}")

    plt.figure(figsize=(10, 10))
    plt.axis("equal")

    result, num = construct_curve(points, weights, nodes, degree, density)

    if split is True:

        X = numpy.split(result[0], num)
        Y = numpy.split(result[1], num)

        for x, y in zip(X, Y):
            plt.plot(x, y, linewidth=3)

    else:
        plt.plot(result[0], result[1], linewidth=3)

    if cascade is True:
        plt.plot(points[:, 0], points[:, 1], color="k", linewidth=0.5)

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

def bershtein_basis(n, i, t):

    C_n_i = numpy.math.factorial(n)/(numpy.math.factorial(i) * numpy.math.factorial(n - i))

    return C_n_i * t ** i * (1 - t) ** (n - i)

def besier_curve(points, weights=None, density=100):
    
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
