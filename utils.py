import torch
import numpy as np
import torch.nn.functional as F


def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)


def save_model(model, binary):
    if binary:
        torch.save(model.state_dict(), "saved_model_binary.pt")
    else:
        torch.save(model.state_dict(), "saved_model.pt")


def load_model(model, device, binary):
    if binary:
        model.load_state_dict(torch.load("saved_model_binary.pt", map_location=device))
    else:
        model.load_state_dict(torch.load("saved_model.pt", map_location=device))
    return model


def get_R_from_angles(angles):
    x, y, z = angles.astype(np.float32)
    # x
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    # y
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    # z
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    return R


def get_R_from_uv(u, v, eps=1e-8):
    e1 = F.normalize(u + eps)
    e2 = F.normalize(v - torch.einsum('bi,bi->b', e1, v).unsqueeze(-1) * e1 + eps)
    e3 = torch.cross(e1, e2, dim=1) + eps
    R = torch.cat([e1.unsqueeze(2), e2.unsqueeze(2), e3.unsqueeze(2)], dim=2)
    return R


def mean_angle_loss(predictions, labels, euler=False):
    if euler:
        M1 = euler_angles_to_matrix(predictions, convention="ZYX")
        M2 = euler_angles_to_matrix(labels, convention="ZYX")
    else:
        M1 = get_R_from_uv(predictions[:, :3], predictions[:, 3:])
        M2 = get_R_from_uv(labels[:, :3], labels[:, 3:])

    loss = get_mean_angle(M1, M2)
    return loss


def get_mean_angle(M1, M2, eps=1e-7):
    dot_product = torch.einsum('bmn,bnm->bm', M1.transpose(1, 2), M2)
    dot_product = torch.clip(dot_product, -1+eps, 1-eps)
    metric = torch.mean(180 / torch.pi * torch.arccos(dot_product))
    return metric


def get_6D_representations(angles):
    R = get_R_from_angles(angles)
    u = R[:, 0]
    v = R[:, 1]
    rep_6D = np.concatenate([u, v])
    return rep_6D


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


if __name__ == "__main__":
    predictions = torch.tensor([[1, 4, 1]], dtype=torch.float)
    labels = torch.tensor([[1, 4, 1]], dtype=torch.float)
    loss = mean_angle_loss(predictions, labels, euler=True)
    print(loss)
    m = get_R_from_angles(np.array([1, 2, 3]))
    print(m)