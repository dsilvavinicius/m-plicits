# coding: utf-8

from copy import deepcopy
import torch
import torch.nn.functional as F
import i3d.diff_operators


def sdf_constraint_on_surf(gt_sdf, pred_sdf):
    return torch.where(gt_sdf == 0, pred_sdf**2, torch.zeros_like(pred_sdf))


def sdf_constraint_off_surf(gt_sdf, pred_sdf):
    return torch.where(
        gt_sdf != 0, (gt_sdf - pred_sdf) ** 2, torch.zeros_like(pred_sdf)
    )


def sdf_constraint_off_surf_pc(gt_sdf, pred_sdf):
    return torch.where(
        gt_sdf != 0, (gt_sdf - torch.abs(pred_sdf)) ** 2, torch.zeros_like(pred_sdf)
    )


def vector_aligment_on_surf(gt_sdf, gt_vectors, pred_vectors):
    return torch.where(
        gt_sdf == 0,
        1 - F.cosine_similarity(pred_vectors, gt_vectors, dim=-1)[..., None],
        torch.zeros_like(gt_sdf),
    )


def direction_aligment_on_surf(gt_sdf, gt_dirs, pred_dirs):
    return torch.where(
        gt_sdf == 0,
        1 - (F.cosine_similarity(pred_dirs, gt_dirs, dim=-1)[..., None]) ** 2,
        torch.zeros_like(gt_sdf),
    )


def eikonal_constraint(gradient):
    return (gradient.norm(dim=-1) - 1.0) ** 2


def off_surface_without_sdf_constraint(gt_sdf, pred_sdf, radius=1e2):
    """
    This function penalizes the pred_sdf of points in gt_sdf!=0
    Used in SIREN's paper
    """
    return torch.where(
        gt_sdf == 0,
        torch.zeros_like(pred_sdf),
        torch.exp(-radius * torch.abs(pred_sdf)),
    )


def on_surface_normal_constraint(gt_sdf, gt_normals, grad):
    """
    This function return a number that measure how far gt_normals
    and grad are aligned in the zero-level set of sdf.
    """
    return torch.where(
        gt_sdf == 0,
        1 - F.cosine_similarity(grad, gt_normals, dim=-1)[..., None],
        torch.zeros_like(grad[..., :1]),
    )


def sdf_sitzmann(X, gt):
    """Loss function employed in Sitzmann et al. for SDF experiments [1].

    Parameters
    ----------
    X: dict[str=>torch.Tensor]
        Model output with the following keys: 'model_in' and 'model_out'
        with the model input and SDF values respectively.

    gt: dict[str=>torch.Tensor]
        Ground-truth data with the following keys: 'sdf' and 'normals', with
        the actual SDF values and the input data normals, respectively.

    Returns
    -------
    loss: dict[str=>torch.Tensor]
        The calculated loss values for each constraint.

    References
    ----------
    [1] Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B.,
    & Wetzstein, G. (2020). Implicit Neural Representations with Periodic
    Activation Functions. ArXiv. Retrieved from http://arxiv.org/abs/2006.09661
    """
    gt_sdf = gt["sdf"]
    gt_normals = gt["normals"]

    coords = X["model_in"]
    pred_sdf = X["model_out"]

    grad = i3d.diff_operators.gradient(pred_sdf, coords)

    # Initial-boundary constraints
    sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(
        gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf))
    )
    normal_constraint = torch.where(
        gt_sdf != -1,
        1 - F.cosine_similarity(grad, gt_normals, dim=-1)[..., None],
        torch.zeros_like(grad[..., :1]),
    )

    # PDE constraints
    grad_constraint = torch.abs(grad.norm(dim=-1) - 1)

    return {
        "sdf_constraint": torch.abs(sdf_constraint).mean() * 3e3,
        "inter_constraint": inter_constraint.mean() * 1e2,
        "normal_constraint": normal_constraint.mean() * 1e2,
        "grad_constraint": grad_constraint.mean() * 5e1,
    }


def true_sdf(X, gt):
    """Uses true SDF value for off-surface points.

    Parameters
    ----------
    X: dict[str=>torch.Tensor]
        Model output with the following keys: 'model_in' and 'model_out'
        with the model input and SDF values respectively.

    gt: dict[str=>torch.Tensor]
        Ground-truth data with the following keys: 'sdf' and 'normals', with
        the actual SDF values and the input data normals, respectively.

    Returns
    -------
    loss: dict[str=>torch.Tensor]
        The calculated loss values for each constraint.
    """
    gt_sdf = gt["sdf"]
    gt_normals = gt["normals"]

    coords = X["model_in"]
    pred_sdf = X["model_out"]

    gradient = i3d.diff_operators.gradient(pred_sdf, coords)

    # Initial-boundary constraints
    sdf_on_surf = sdf_constraint_on_surf(gt_sdf, pred_sdf)
    sdf_off_surf = sdf_constraint_off_surf(gt_sdf, pred_sdf)
    # sdf_off_surf = off_surface_without_sdf_constraint(gt_sdf, pred_sdf)
    normal_constraint = vector_aligment_on_surf(gt_sdf, gt_normals, gradient)

    # PDE constraints
    grad_constraint = eikonal_constraint(gradient).unsqueeze(-1)

    return {
        "sdf_on_surf": sdf_on_surf.mean() * 3e3,
        "sdf_off_surf": sdf_off_surf.mean() * 2e2,
        "normal_constraint": normal_constraint.mean() * 1e2,  # 1e1,
        "grad_constraint": grad_constraint.mean() * 5e1,  # 1e1
    }


def true_sdf_pc(X, gt):
    """Uses true SDF value for off-surface points.

    Parameters
    ----------
    X: dict[str=>torch.Tensor]
        Model output with the following keys: 'model_in' and 'model_out'
        with the model input and SDF values respectively.

    gt: dict[str=>torch.Tensor]
        Ground-truth data with the following keys: 'sdf' and 'normals', with
        the actual SDF values and the input data normals, respectively.

    Returns
    -------
    loss: dict[str=>torch.Tensor]
        The calculated loss values for each constraint.
    """
    gt_sdf = gt["sdf"]
    gt_normals = gt["normals"]

    coords = X["model_in"]
    pred_sdf = X["model_out"]

    gradient = i3d.diff_operators.gradient(pred_sdf, coords)

    # Initial-boundary constraints
    sdf_on_surf = sdf_constraint_on_surf(gt_sdf, pred_sdf)
    sdf_off_surf = sdf_constraint_off_surf_pc(gt_sdf, pred_sdf)
    # sdf_off_surf = off_surface_without_sdf_constraint(gt_sdf, pred_sdf)
    normal_constraint = vector_aligment_on_surf(gt_sdf, gt_normals, gradient)

    # PDE constraints
    grad_constraint = eikonal_constraint(gradient).unsqueeze(-1)

    return {
        "sdf_on_surf": sdf_on_surf.mean() * 3e3,
        "sdf_off_surf": sdf_off_surf.mean() * 2e2,
        "normal_constraint": normal_constraint.mean() * 1e2,  # 1e1,
        "grad_constraint": grad_constraint.mean() * 5e1,  # 1e1
    }


def loss_sdf_on_neighborhood(X, gt):
    """Uses true SDF value for off-surface points.

    Parameters
    ----------
    X: dict[str=>torch.Tensor]
        Model output with the following keys: 'model_in' and 'model_out'
        with the model input and SDF values respectively.

    gt: dict[str=>torch.Tensor]
        Ground-truth data with the following keys: 'sdf' and 'normals', with
        the actual SDF values and the input data normals, respectively.

    Returns
    -------
    loss: dict[str=>torch.Tensor]
        The calculated loss values for each constraint.
    """
    gt_sdf = gt["sdf"]
    gt_normals = gt["normals"]

    coords = X["model_in"]
    pred_sdf = X["model_out"]

    gradient = i3d.diff_operators.gradient(pred_sdf, coords)

    # Initial-boundary constraints
    # sdf_constraint = (gt_sdf - pred_sdf)**2
    sdf_constraint = torch.where(
        gt_sdf != -1, (gt_sdf - pred_sdf) ** 2, torch.zeros_like(pred_sdf)
    )

    normal_constraint = torch.where(
        gt_sdf != -1,
        1 - F.cosine_similarity(gt_normals, gradient, dim=-1)[..., None],
        torch.zeros_like(gt_sdf),
    )

    inter_constraint = torch.where(
        gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e3 * (pred_sdf) ** 2)
    )

    # PDE constraints
    grad_constraint = eikonal_constraint(gradient).unsqueeze(-1)

    return {
        "sdf_constraint": sdf_constraint.mean() * 3e3,
        "normal_constraint": normal_constraint.mean() * 2e2,  # 1e1,
        "inter_constraint": inter_constraint.mean(),
        "grad_constraint": grad_constraint.mean() * 5e1,  # 1e1
    }


# learn a second model that add details to a previously trained model
class loss_add_detail(torch.nn.Module):
    DEFAULT_WEIGHTS = {
        "sdf_constraint": 3e3,
        "normal_constraint": 1e2,  # 1e1,
        "grad_constraint": 5e1,  # 1e1,
    }

    def __init__(self, model_coarse, constraint_weights: dict = dict()):
        super().__init__()
        # Define the model.
        self.model_coarse = model_coarse.eval()
        if not constraint_weights or constraint_weights is None:
            self.constraint_weights = deepcopy(loss_add_detail.DEFAULT_WEIGHTS)
        else:
            self.constraint_weights = dict(
                (k, float(v)) for k, v in constraint_weights.items()
            )

    def forward(self, X, gt):
        gt_sdf = gt["sdf"]
        gt_normals = gt["normals"]

        coords = X["model_in"]
        pred_sdf = X["model_out"]

        gradient = i3d.diff_operators.gradient(pred_sdf, coords)

        coarse = self.model_coarse(coords)
        coarse_coords = coarse["model_in"]
        coarse_sdf = coarse["model_out"]
        coarse_gradient = i3d.diff_operators.gradient(
            coarse_sdf, coarse_coords
        ).detach()

        # Initial-boundary constraints
        sdf_constraint = torch.where(
            gt_sdf != -1,
            (gt_sdf - pred_sdf - coarse_sdf) ** 2,
            torch.zeros_like(pred_sdf),
        )

        normal_constraint = torch.where(
            gt_sdf != -1,
            1
            - F.cosine_similarity(gt_normals, gradient + coarse_gradient, dim=-1)[
                ..., None
            ],
            torch.zeros_like(gt_sdf),
        )

        # PDE constraints
        grad_constraint = eikonal_constraint(gradient + coarse_gradient).unsqueeze(-1)

        csdf = self.constraint_weights["sdf_constraint"]
        cnormal = self.constraint_weights["normal_constraint"]
        cgrad = self.constraint_weights["grad_constraint"]
        return {
            "sdf_constraint": sdf_constraint.mean() * csdf,
            "normal_constraint": normal_constraint.mean() * cnormal,
            "grad_constraint": grad_constraint.mean() * cgrad,
        }


# learn a second model that add details to a previously trained model
class loss_add_detail_fine(torch.nn.Module):
    DEFAULT_WEIGHTS = {
        "sdf_constraint": 3e3,
        "normal_constraint": 1e2,  # 1e1,
        "grad_constraint": 5e1,  # 1e1,
    }

    def __init__(self, model_coarse, res_model, constraint_weights: dict = dict()):
        super().__init__()
        # Define the model.
        self.model_coarse = model_coarse.eval()
        self.res_model = res_model.eval()

        if not constraint_weights or constraint_weights is None:
            self.constraint_weights = deepcopy(loss_add_detail_fine.DEFAULT_WEIGHTS)
        else:
            self.constraint_weights = dict(
                (k, float(v)) for k, v in constraint_weights.items()
            )

    def forward(self, X, gt):
        gt_sdf = gt["sdf"]
        gt_normals = gt["normals"]

        coords = X["model_in"]
        pred_sdf = X["model_out"]

        gradient = i3d.diff_operators.gradient(pred_sdf, coords)

        # coarse
        coarse = self.model_coarse(coords)
        coarse_coords = coarse["model_in"]
        coarse_sdf = coarse["model_out"]
        coarse_gradient = i3d.diff_operators.gradient(
            coarse_sdf, coarse_coords
        ).detach()

        # res
        res = self.res_model(coords)
        res_coords = res["model_in"]
        res_sdf = res["model_out"]
        res_gradient = i3d.diff_operators.gradient(res_sdf, res_coords).detach()

        # Initial-boundary constraints
        sdf_constraint = torch.where(
            gt_sdf != -1,
            (gt_sdf - pred_sdf - res_sdf - coarse_sdf) ** 2,
            torch.zeros_like(pred_sdf),
        )

        pred_grads = gradient + res_gradient + coarse_gradient
        normal_constraint = torch.where(
            gt_sdf != -1,
            1 - F.cosine_similarity(gt_normals, pred_grads, dim=-1)[..., None],
            torch.zeros_like(gt_sdf),
        )

        # PDE constraints
        grad_constraint = eikonal_constraint(pred_grads).unsqueeze(-1)

        csdf = self.constraint_weights["sdf_constraint"]
        cnormal = self.constraint_weights["normal_constraint"]
        cgrad = self.constraint_weights["grad_constraint"]
        return {
            "sdf_constraint": sdf_constraint.mean() * csdf,
            "normal_constraint": normal_constraint.mean() * cnormal,
            "grad_constraint": grad_constraint.mean() * cgrad,
        }


class loss_add_detail_sitzmann(torch.nn.Module):
    DEFAULT_WEIGHTS = {
        "sdf_constraint": 3e3,
        "inter_constraint": 1e2,
        "normal_constraint": 1e2,
        "grad_constraint": 5e1,
    }

    def __init__(self, model_coarse, constraint_weights: dict = dict()):
        super().__init__()
        # Define the model.
        self.model_coarse = model_coarse.eval()
        if not constraint_weights or constraint_weights is None:
            self.constraint_weights = deepcopy(loss_add_detail_sitzmann.DEFAULT_WEIGHTS)
        else:
            self.constraint_weights = dict(
                (k, float(v)) for k, v in constraint_weights.items()
            )

    def forward(self, X, gt):
        gt_sdf = gt["sdf"]
        gt_normals = gt["normals"]

        coords = X["model_in"]
        pred_sdf = X["model_out"]

        gradient = i3d.diff_operators.gradient(pred_sdf, coords)

        coarse = self.model_coarse(coords)
        coarse_coords = coarse["model_in"]
        coarse_sdf = coarse["model_out"]
        coarse_gradient = i3d.diff_operators.gradient(
            coarse_sdf, coarse_coords
        ).detach()

        # Initial-boundary constraints
        sdf_constraint = torch.where(
            gt_sdf != -1,
            (gt_sdf - pred_sdf - coarse_sdf) ** 2,
            torch.zeros_like(pred_sdf),
        )
        inter_constraint = torch.where(
            gt_sdf != -1,
            torch.zeros_like(pred_sdf),
            torch.exp(-1e2 * torch.abs(gt_sdf - pred_sdf - coarse_sdf)),
        )
        normal_constraint = torch.where(
            gt_sdf != -1,
            1
            - F.cosine_similarity(gt_normals, gradient + coarse_gradient, dim=-1)[
                ..., None
            ],
            torch.zeros_like(gt_sdf),
        )

        # PDE constraints
        grad_constraint = eikonal_constraint(gradient + coarse_gradient).unsqueeze(-1)

        csdf = self.constraint_weights["sdf_constraint"]
        cinter = self.constraint_weights["inter_constraint"]
        cnormal = self.constraint_weights["normal_constraint"]
        cgrad = self.constraint_weights["grad_constraint"]
        return {
            "sdf_constraint": sdf_constraint.mean() * csdf,
            "inter_constraint": inter_constraint.mean() * cinter,
            "normal_constraint": normal_constraint.mean() * cnormal,
            "grad_constraint": grad_constraint.mean() * cgrad,
        }


class loss_add_detail_sitzmann_fine(torch.nn.Module):
    DEFAULT_WEIGHTS = {
        "sdf_constraint": 3e3,
        "inter_constraint": 1e2,
        "normal_constraint": 1e2,
        "grad_constraint": 5e1,
    }

    def __init__(self, model_coarse, model_medium, constraint_weights: dict = dict()):
        super().__init__()
        # Define the model.
        self.model_coarse = model_coarse.eval()
        self.model_medium = model_medium.eval()

        if not constraint_weights or constraint_weights is None:
            self.constraint_weights = deepcopy(
                loss_add_detail_sitzmann_fine.DEFAULT_WEIGHTS
            )
        else:
            self.constraint_weights = dict(
                (k, float(v)) for k, v in constraint_weights.items()
            )

    def forward(self, X, gt):
        gt_sdf = gt["sdf"]
        gt_normals = gt["normals"]

        coords = X["model_in"]
        pred_sdf = X["model_out"]

        gradient = i3d.diff_operators.gradient(pred_sdf, coords)

        # coarse
        coarse = self.model_coarse(coords)
        coarse_coords = coarse["model_in"]
        coarse_sdf = coarse["model_out"]
        coarse_gradient = i3d.diff_operators.gradient(
            coarse_sdf, coarse_coords
        ).detach()

        # res
        res = self.model_medium(coords)
        res_coords = res["model_in"]
        res_sdf = res["model_out"]
        res_gradient = i3d.diff_operators.gradient(res_sdf, res_coords).detach()

        # Initial-boundary constraints
        sdf_constraint = torch.where(
            gt_sdf != -1,
            (gt_sdf - pred_sdf - res_sdf - coarse_sdf) ** 2,
            torch.zeros_like(pred_sdf),
        )
        inter_constraint = torch.where(
            gt_sdf != -1,
            torch.zeros_like(pred_sdf),
            torch.exp(-1e2 * torch.abs(gt_sdf - pred_sdf - res_sdf - coarse_sdf)),
        )

        pred_grads = gradient + res_gradient + coarse_gradient
        normal_constraint = torch.where(
            gt_sdf != -1,
            1 - F.cosine_similarity(gt_normals, pred_grads, dim=-1)[..., None],
            torch.zeros_like(gt_sdf),
        )

        # PDE constraints
        grad_constraint = eikonal_constraint(pred_grads).unsqueeze(-1)

        csdf = self.constraint_weights["sdf_constraint"]
        cinter = self.constraint_weights["inter_constraint"]
        cnormal = self.constraint_weights["normal_constraint"]
        cgrad = self.constraint_weights["grad_constraint"]
        return {
            "sdf_constraint": sdf_constraint.mean() * csdf,
            "inter_constraint": inter_constraint.mean() * cinter,
            "normal_constraint": normal_constraint.mean() * cnormal,
            "grad_constraint": grad_constraint.mean() * cgrad,
        }


class loss_add_detail_end2end_idf(torch.nn.Module):
    DEFAULT_WEIGHTS = {
        "sdf_constraint": 3e3,
        "inter_constraint": 1e2,
        "normal_constraint": 1e2,
        "grad_constraint": 5e1,
    }

    def __init__(self, steps_base_only, constraint_weights: dict = dict()):
        super().__init__()
        # Define the model.
        # self.model_coarse = model_coarse.eval()
        self.steps_base_only = steps_base_only
        if not constraint_weights or constraint_weights is None:
            self.constraint_weights = deepcopy(loss_add_detail_sitzmann.DEFAULT_WEIGHTS)
        else:
            self.constraint_weights = dict(
                (k, float(v)) for k, v in constraint_weights.items()
            )

    def forward(self, X, gt, curr_step):
        gt_sdf = gt["sdf"]
        gt_normals = gt["normals"]

        coords = X["model_in"]
        pred_sdf = X["model_out"]

        gradient = i3d.diff_operators.gradient(pred_sdf, coords)

        coarse = self.model_coarse(coords)
        coarse_coords = coarse["model_in"]
        coarse_sdf = coarse["model_out"]
        coarse_gradient = i3d.diff_operators.gradient(
            coarse_sdf, coarse_coords
        ).detach()

        # Initial-boundary constraints
        sdf_constraint = torch.where(
            gt_sdf != -1,
            (gt_sdf - pred_sdf - coarse_sdf) ** 2,
            torch.zeros_like(pred_sdf),
        )
        inter_constraint = torch.where(
            gt_sdf != -1,
            torch.zeros_like(pred_sdf),
            torch.exp(-1e2 * torch.abs(gt_sdf - pred_sdf - coarse_sdf)),
        )
        normal_constraint = torch.where(
            gt_sdf != -1,
            1
            - F.cosine_similarity(gt_normals, gradient + coarse_gradient, dim=-1)[
                ..., None
            ],
            torch.zeros_like(gt_sdf),
        )

        # PDE constraints
        grad_constraint = eikonal_constraint(gradient + coarse_gradient).unsqueeze(-1)

        csdf = self.constraint_weights["sdf_constraint"]
        cinter = self.constraint_weights["inter_constraint"]
        cnormal = self.constraint_weights["normal_constraint"]
        cgrad = self.constraint_weights["grad_constraint"]
        return {
            "sdf_constraint": sdf_constraint.mean() * csdf,
            "inter_constraint": inter_constraint.mean() * cinter,
            "normal_constraint": normal_constraint.mean() * cnormal,
            "grad_constraint": grad_constraint.mean() * cgrad,
        }
    

def principal_directions_sdf(model_output, gt):
    """Uses true SDF value off surface and tries to align the principal
    directions on the 0 level-set.

    Parameters
    ----------
    X: dict[str=>torch.Tensor]
        Model output with the following keys: 'model_in' and 'model_out'
        with the model input and SDF values respectively.

    gt: dict[str=>torch.Tensor]
        Ground-truth data with the following keys: 'sdf', 'normals',
        'min_curvatures', 'max_curvatures', 'max_principal_directions', with
        the actual SDF values, the input data normals, minimum and maximum
        curvatures, and maximal principal directions, respectively.

    Returns
    -------
    loss: dict[str=>torch.Tensor]
        The calculated loss values for each constraint.
    """
    gt_sdf = gt["sdf"]
    gt_normals = gt["normals"]
    gt_min_curvature = gt["min_curvatures"]
    gt_max_curvature = gt["max_curvatures"]
    gt_dirs = gt["max_principal_directions"]

    coords = model_output["model_in"]
    pred_sdf = model_output["model_out"]

    gradient = i3d.diff_operators.gradient(pred_sdf, coords)
    hessian = i3d.diff_operators.hessian(pred_sdf, coords)
    pred_dirs = i3d.diff_operators.principal_directions(gradient, hessian)

    dirs_constraint = direction_aligment_on_surf(gt_sdf, gt_dirs, pred_dirs[0][..., :3])

    aux_dirs_constraint = torch.where(
        gt_sdf == 0,
        F.cosine_similarity(pred_dirs[0][..., :3], gt_normals, dim=-1)[..., None] ** 2,
        torch.zeros_like(gt_sdf),
    )

    dirs_constraint = dirs_constraint + 0.1 * aux_dirs_constraint

    # Removing problematic curvatures and planar points
    planar_curvature = 0.5 * torch.abs(gt_min_curvature - gt_max_curvature)
    dirs_constraint = torch.where(
        planar_curvature > 10, dirs_constraint, torch.zeros_like(dirs_constraint)
    )
    dirs_constraint = torch.where(
        planar_curvature < 5000, dirs_constraint, torch.zeros_like(dirs_constraint)
    )

    return {
        "sdf_on_surf": sdf_constraint_on_surf(gt_sdf, pred_sdf).mean() * 3e3,
        "sdf_off_surf": sdf_constraint_off_surf(gt_sdf, pred_sdf).mean() * 2e2,
        "normal_constraint": vector_aligment_on_surf(
            gt_sdf, gt_normals, gradient
        ).mean()
        * 1e2,  # 1e1,
        "grad_constraint": eikonal_constraint(gradient).mean() * 5e1,
        "dirs_constraint": dirs_constraint.mean(),
    }


def mean_curvature_sdf(model_output, gt):
    """Uses true SDF value off surface and tries to fit the mean curvatures
    on the 0 level-set.

    Parameters
    ----------
    X: dict[str=>torch.Tensor]
        Model output with the following keys: 'model_in' and 'model_out'
        with the model input and SDF values respectively.

    gt: dict[str=>torch.Tensor]
        Ground-truth data with the following keys: 'sdf', 'normals', and
        'curvature' with the actual SDF values, the input data normals, and
        gaussian curvatures, respectively.

    Returns
    -------
    loss: dict[str=>torch.Tensor]
        The calculated loss values for each constraint.
    """
    gt_sdf = gt["sdf"]
    gt_normals = gt["normals"]
    gt_curvature = gt["curvature"]

    coords = model_output["model_in"]
    pred_sdf = model_output["model_out"]

    gradient = i3d.diff_operators.gradient(pred_sdf, coords)

    # mean curvature
    pred_curvature = i3d.diff_operators.divergence(gradient, coords)
    curv_constraint = torch.where(
        gt_sdf == 0,
        (pred_curvature - gt_curvature) ** 2,
        torch.zeros_like(pred_curvature),
    )

    # Wherever boundary_values is not equal to zero, we interpret it as a
    # boundary constraint.
    return {
        "sdf_on_surf": sdf_constraint_on_surf(gt_sdf, pred_sdf).mean() * 3e3,
        "sdf_off_surf": sdf_constraint_off_surf(gt_sdf, pred_sdf).mean() * 2e2,
        "normal_constraint": vector_aligment_on_surf(
            gt_sdf, gt_normals, gradient
        ).mean()
        * 1e2,  # 1e1,
        "grad_constraint": eikonal_constraint(gradient).unsqueeze(-1).mean() * 5e1,
        "curv_constraint": curv_constraint.mean() * 1e-1,
    }
