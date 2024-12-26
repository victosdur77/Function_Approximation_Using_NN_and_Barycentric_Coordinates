import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.stats import entropy
import scipy.sparse as sparse
from ripser import ripser

# function for calculate intermediate points using baricentric coordinates
def generate_baricentric_points(points, num_intermediate):
    """
    Generates intermediate points between consecutive pairs in the `points` tensor using barycentric coordinates.
    """
    intermediate_points = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        # Generate intermediate points
        for j in range(1, num_intermediate + 1):
            t = j / (num_intermediate + 1)  # Barycentric proportion
            point = (1 - t) * p0 + t * p1  # Barycentric formula
            intermediate_points.append(point)
    
    return torch.stack(intermediate_points)
    
# function for calculate persistence diagramas using LowerStar filtration (not torch).
def calculatePersistenceDiagrams_LowerStar(t,x):
    N = x.shape[0]
    I = np.arange(N-1)
    J = np.arange(1, N)
    V = np.maximum(x[0:-1], x[1::])
    # Add vertex birth times along the diagonal of the distance matrix
    I = np.concatenate((I, np.arange(N)))
    J = np.concatenate((J, np.arange(N)))
    V = np.concatenate((V, x))
    #Create the sparse distance matrix
    D = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    dgms = ripser(D, maxdim=3, distance_matrix=True)['dgms'] # doesn't matter the maxdim as there is only diagram for dimension 0 in the lowerstar filtration.
    return dgms

# function for obtain persistence diagrama of specific dimension.
def obtainDiagramDimension(Diagrams,dimension):
    dgm=Diagrams[dimension]
    dgm = dgm[dgm[:, 1]-dgm[:, 0] > 1e-3, :]
    return dgm

# function for remove infinity values for persistence diagram.
def limitDiagramLowerStar(Diagram,maximumFiltration):
    infinity_mask = np.isinf(Diagram)
    Diagram[infinity_mask] = maximumFiltration + 1
    return Diagram

# function for compute PE from persistence barcode
def computePersistenceEntropy(persistentBarcode):
    l=[]
    for i in persistentBarcode:
        l.append(i[1]-i[0])
    L = sum(l)
    p=l/L
    entropia=-np.sum(p*np.log(p))
    return entropia # round(entropia,4)

# persistent entropy calculation in pytorch. use log base 2 in torch_topological code
def persistent_entropyTorch(D, **kwargs):
    """Calculate persistent entropy of a persistence diagram.

    Parameters
    ----------
    D : `torch.tensor`
        Persistence diagram, assumed to be in shape `(n, 2)`, where each
        entry corresponds to a tuple of the form :math:`(x, y)`, with
        :math:`x` denoting the creation of a topological feature and
        :math:`y` denoting its destruction.

    Returns
    -------
    Persistent entropy of `D`.
    """
    persistence = torch.diff(D)
    persistence = persistence[torch.isfinite(persistence)].abs()

    P = persistence.sum()
    probabilities = persistence / P

    # Ensures that a probability of zero will just result in
    # a logarithm of zero as well. This is required whenever
    # one deals with entropy calculations.
    indices = probabilities > 0
    log_prob = torch.zeros_like(probabilities)
    log_prob[indices] = torch.log(probabilities[indices])

    return -torch.sum(probabilities * log_prob)

# PyTorch module for calculating differentiable Lower Star persistence diagrams.
class LowerStarPersistence(nn.Module):
    def __init__(self, maxdim=0):
        """
        PyTorch module for calculating differentiable Lower Star persistence diagrams. param maxdim: Maximum dimension to calculate the diagram (default 0).
        """
        super(LowerStarPersistence, self).__init__()
        self.maxdim = maxdim

    def forward(self, x):
        """
        Calculates the Lower Star persistence diagram in a differentiable form.
        param x: 1D tensor of filtration values (shape: N).
        return: Persistence diagram as tensor.
        """
        return LowerStarFunction.apply(x, self.maxdim)

# PyTorch module for calculating differentiable Lower Star persistence diagrams.
class LowerStarFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, maxdim):
        """
        Calculate the persistence diagram using Ripser.
        param x: 1D tensor of filtration values.
        param maxdim: Maximum dimension for Ripser.
        return: Persistence diagram as tensor.
        """
        N = x.shape[0]

        # Build indices for edges of the Lower Star Complex
        I = torch.arange(N - 1, device=x.device) # Source indices
        J = torch.arange(1, N, device=x.device) # Destination indices
        V = torch.maximum(x[:-1], x[1:]) # Filter by maximum between consecutive points

        # Adding birth times for vertices (diagonal of the matrix)
        I = torch.cat([I, torch.arange(N, device=x.device)])
        J = torch.cat([J, torch.arange(N, device=x.device)])
        V = torch.cat([V, x])

        # Create sparse matrix using SciP
        I = I.detach().cpu().numpy()
        J = J.detach().cpu().numpy()
        V = V.detach().cpu().numpy()
        D = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()

        # Calculating the persistence diagram using Ripser
        dgms = ripser(D, maxdim=maxdim, distance_matrix=True)['dgms']

        # Convert diagram to tensor and save for backpropagation.
        dgm_0 = dgms[0]
        dgm_tensor = torch.tensor(dgm_0, dtype=torch.float32, device=x.device)
        ctx.save_for_backward(x)
        ctx.dgm_tensor = dgm_tensor
        return dgm_tensor

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backpropagation for the relationship between x and the persistence diagram.
        """
        x, = ctx.saved_tensors
        dgm_tensor = ctx.dgm_tensor

        # Approximate gradients: relating changes in filtration to persistencies
        grad = torch.zeros_like(x)
        for i, (birth, death) in enumerate(dgm_tensor):
            # Simple difference (gradient proxy)
            grad += grad_output[i, 0] * (x - birth) - grad_output[i, 1] * (x - death)
        return grad, None

# persistent entropy calculation in pytorch, valid as loss function.
class PersistentEntropyLossTorch(torch.nn.Module):
    r"""Implement loss based on summary statistic.

    This is a generic loss function based on topological summary
    statistics. It implements a loss of the following form:

    .. math:: \|PE(X) - PE(Y)\|^p

    In the preceding equation, `PE` refers to persistent entropy based on persistence diagrams.. For example
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.p = kwargs.get("p", 1.0)
        self.kwargs = kwargs
        self.stat_fn = persistent_entropyTorch
        
    def forward(self, X, Y=None):
        r"""Calculate loss based on input tensor(s).

        Parameters
        ----------
        X : list of :class:`PersistenceInformation`
            Source information. Supposed to contain persistence diagrams
            and persistence pairings.

        Y : list of :class:`PersistenceInformation` or `None`
            Optional target information. If set, evaluates a difference
            in loss functions as shown in the introduction. If `None`,
            a simpler variant of the loss will be evaluated.

        Returns
        -------
        torch.tensor
            Given a statistic(persistent entropy= :math:`s`, the function returns the
            following expression:

            .. math:: \|s(X) - s(Y)\|^p

            In case no target tensor `Y` has been provided, the latter part
            of the expression amounts to `0`.
        """
        stat_src = self._evaluate_stat_fn(X)

        if Y is not None:
            stat_target = self._evaluate_stat_fn(Y)
            return (stat_target - stat_src).abs().pow(self.p)
        else:
            return stat_src.abs().pow(self.p)

    def _evaluate_stat_fn(self, X):
        """Evaluate statistic function for a given tensor."""
        return torch.sum(
            torch.stack(
                [
                    self.stat_fn(X)
                ]
            )
        )

# function for plot signal data and his persistent diagram of specific dimension.
def plotSignal_PersistentDiagram(t,signal,dimension):
    dgms = calculatePersistenceDiagrams_LowerStar(t,signal)    
    dgm0 = obtainDiagramDimension(dgms,dimension)
    allgrid = np.unique(dgm0.flatten())
    allgrid = allgrid[allgrid < np.inf]
    xs = np.unique(dgm0[:, 0])
    ys = np.unique(dgm0[:, 1])
    ys = ys[ys < np.inf]

    #Plot the time series and the persistence diagram
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(t, signal,'-o')
    ax = plt.gca()
    ax.set_yticks(allgrid)
    ax.set_xticks([])
    plt.grid(linewidth=1, linestyle='--')
    plt.title("Señal")
    plt.xlabel("t")

    plt.subplot(122)
    ax = plt.gca()
    ax.set_yticks(ys)
    ax.set_xticks(xs)
    plt.grid(linewidth=1, linestyle='--')
    plot_diagrams(dgm0, size=50)
    plt.title(f"Persistence Diagram, dimension = {dimension}")
    plt.show()

# function for plot peristence diagrams.
def plot_diagrams(
    diagrams,
    plot_only=None,
    title=None,
    xy_range=None,
    labels=None,
    colormap="default",
    size=20,
    ax_color=np.array([0.0, 0.0, 0.0]),
    diagonal=True,
    lifetime=False,
    legend=True,
    show=False,
    ax=None
):
    """A helper function to plot persistence diagrams. 

    Parameters
    ----------
    diagrams: ndarray (n_pairs, 2) or list of diagrams
        A diagram or list of diagrams. If diagram is a list of diagrams, 
        then plot all on the same plot using different colors.
    plot_only: list of numeric
        If specified, an array of only the diagrams that should be plotted.
    title: string, default is None
        If title is defined, add it as title of the plot.
    xy_range: list of numeric [xmin, xmax, ymin, ymax]
        User provided range of axes. This is useful for comparing 
        multiple persistence diagrams.
    labels: string or list of strings
        Legend labels for each diagram. 
        If none are specified, we use H_0, H_1, H_2,... by default.
    colormap: string, default is 'default'
        Any of matplotlib color palettes. 
        Some options are 'default', 'seaborn', 'sequential'. 
        See all available styles with

        .. code:: python

            import matplotlib as mpl
            print(mpl.styles.available)

    size: numeric, default is 20
        Pixel size of each point plotted.
    ax_color: any valid matplotlib color type. 
        See [https://matplotlib.org/api/colors_api.html](https://matplotlib.org/api/colors_api.html) for complete API.
    diagonal: bool, default is True
        Plot the diagonal x=y line.
    lifetime: bool, default is False. If True, diagonal is turned to False.
        Plot life time of each point instead of birth and death. 
        Essentially, visualize (x, y-x).
    legend: bool, default is True
        If true, show the legend.
    show: bool, default is False
        Call plt.show() after plotting. If you are using self.plot() as part 
        of a subplot, set show=False and call plt.show() only once at the end.
    """

    ax = ax or plt.gca()
    plt.style.use(colormap)

    xlabel, ylabel = "Birth", "Death"

    if not isinstance(diagrams, list):
        # Must have diagrams as a list for processing downstream
        diagrams = [diagrams]

    if labels is None:
        # Provide default labels for diagrams if using self.dgm_
        labels = ["$H_{{{}}}$".format(i) for i , _ in enumerate(diagrams)]

    if plot_only:
        diagrams = [diagrams[i] for i in plot_only]
        labels = [labels[i] for i in plot_only]

    if not isinstance(labels, list):
        labels = [labels] * len(diagrams)

    # Construct copy with proper type of each diagram
    # so we can freely edit them.
    diagrams = [dgm.astype(np.float32, copy=True) for dgm in diagrams]

    # find min and max of all visible diagrams
    concat_dgms = np.concatenate(diagrams).flatten()
    has_inf = np.any(np.isinf(concat_dgms))
    finite_dgms = concat_dgms[np.isfinite(concat_dgms)]

    # clever bounding boxes of the diagram
    if not xy_range:
        # define bounds of diagram
        ax_min, ax_max = np.min(finite_dgms), np.max(finite_dgms)
        x_r = ax_max - ax_min

        # Give plot a nice buffer on all sides.
        # ax_range=0 when only one point,
        buffer = 1 if xy_range == 0 else x_r / 5

        x_down = ax_min - buffer / 2
        x_up = ax_max + buffer

        y_down, y_up = x_down, x_up
    else:
        x_down, x_up, y_down, y_up = xy_range

    yr = y_up - y_down

    if lifetime:

        # Don't plot landscape and diagonal at the same time.
        diagonal = False

        # reset y axis so it doesn't go much below zero
        y_down = -yr * 0.05
        y_up = y_down + yr

        # set custom ylabel
        ylabel = "Lifetime"

        # set diagrams to be (x, y-x)
        for dgm in diagrams:
            dgm[:, 1] -= dgm[:, 0]

        # plot horizon line
        ax.plot([x_down, x_up], [0, 0], c=ax_color)

    # Plot diagonal
    if diagonal:
        ax.plot([x_down, x_up], [x_down, x_up], "--", c=ax_color)

    # Plot inf line
    if has_inf:
        # put inf line slightly below top
        b_inf = y_down + yr * 0.95
        ax.plot([x_down, x_up], [b_inf, b_inf], "--", c="k", label=r"$\infty$")

        # convert each inf in each diagram with b_inf
        for dgm in diagrams:
            dgm[np.isinf(dgm)] = b_inf

    # Plot each diagram
    for dgm, label in zip(diagrams, labels):

        # plot persistence pairs
        ax.scatter(dgm[:, 0], dgm[:, 1], size, label=label, edgecolor="none")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax.set_xlim([x_down, x_up])
    ax.set_ylim([y_down, y_up])
    ax.set_aspect('equal', 'box')

    if title is not None:
        ax.set_title(title)

    if legend is True:
        ax.legend(loc="lower right")

    if show is True:
        plt.show()