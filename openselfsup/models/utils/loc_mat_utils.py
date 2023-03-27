import torch


def _upcast(t: torch.Tensor) -> torch.Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


def cal_intersection(boxes1, boxes2, h_step_1, w_step_1):
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]

    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    inter = inter / h_step_1 / w_step_1
    return inter


def cal_intersection_batch(boxes1: torch.Tensor, boxes2: torch.Tensor):
    lt = torch.max(boxes1[:, :, None, :2], boxes2[:, None, :, :2])  # [b,N,M,2]
    rb = torch.min(boxes1[:, :, None, 2:], boxes2[:, None, :, 2:])  # [b,N,M,2]

    wh = _upcast(rb - lt).clamp(min=0)  # [b,N,M,2]
    inter = wh[:, :, :, 0] * wh[:, :, :, 1]  # [b,N,M]

    hw_step_1 = boxes1[:, :, 2:] - boxes1[:, :, :2]
    inter1 = inter / (hw_step_1[:, :, [0]] * hw_step_1[:, :, [1]])

    hw_step_2 = boxes2[:, :, 2:] - boxes2[:, :, :2]
    inter2 = inter.transpose(1, 2) / (hw_step_2[:, :, [0]] * hw_step_2[:, :, [1]])

    # print(torch.sum(inter1, dim=-1)[0, ...])
    # print(torch.sum(inter2, dim=-1)[0, ...])
    return inter1, inter2