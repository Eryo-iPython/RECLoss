import torch


#  input (B, HW, 2)
def eight_point(pts1, pts2, M):

    B, N, C = pts1.shape

    pts1_scaled = pts1/M
    pts2_scaled = pts2/M


    #row assignment
    # A_f = torch.zeros((B, pts1_scaled.shape[1], 9))
    #
    # for i in range(pts1_scaled.shape[1]):
    #
    #     point = [pts2_scaled[:, i, 0]*pts1_scaled[:, i, 0],
    #                  pts2_scaled[:, i, 0]*pts1_scaled[:, i, 1],
    #                  pts2_scaled[:, i, 0],
    #                  pts2_scaled[:, i, 1]*pts1_scaled[:, i, 0],
    #                  pts2_scaled[:, i, 1]*pts1_scaled[:, i, 1],
    #                  pts2_scaled[:, i, 1],
    #                  pts1_scaled[:, i, 0],
    #                  pts1_scaled[:, i, 1],
    #                  1.]
    #     # t = np.array(t)
    #     for j in range(9):
    #         A_f[:, i, j] = point[j]

    #parallel computing
    p1 = pts1_scaled.unsqueeze(2)
    p2 = pts2_scaled.unsqueeze(2)
    b, n, r, c = p1.shape
    p1 = torch.cat([p1, torch.ones((b, n, r, 1)).to(p1.device)], dim=-1)
    p2 = torch.cat([p2, torch.ones((b, n, r, 1)).to(p2.device)], dim=-1)

    all_v = p2.transpose(-1, -2)@p1
    A_f = all_v.reshape(b, n, -1)


    u, s, vh = torch.linalg.svd(A_f)
    v = vh.transpose(-2, -1)
    f = v[:, :, -1].reshape(B, 3, 3)

    diag = torch.tensor([1/M, 1/M, 1], requires_grad=False)

    T = torch.diag(diag)

    unscaled_F = torch.einsum('ik, bij, jt->bkt', T, f, T)

    return unscaled_F

def get_matrix(p1, flow):
    b, c, h, w = flow.shape
    M = max(h, w)
    p2 = p1 + flow

    p1 = p1.reshape((b, c, -1))
    p2 = p2.reshape((b, c, -1))

    p1 = p1.permute(0, 2, 1).contiguous()
    p2 = p2.permute(0, 2, 1).contiguous()

    matrix = eight_point(p1, p2, M)

    return matrix, p1

def fun_loss(p1, matrix, pre_flow):
    b, c, h, w = pre_flow.shape

    pre_flow = pre_flow.reshape((b, c, -1))
    pre_flow = pre_flow.permute(0, 2, 1).contiguous()

    pre_p2 = pre_flow + p1

    p1 = torch.cat([p1, torch.ones((b, h*w, 1)).to(p1.device)], dim=-1)
    p2 = torch.cat([pre_p2, torch.ones((b, h*w, 1)).to(pre_flow.device)], dim=-1)

    loss = torch.einsum('nij, njk, ntk->nit', p2, matrix.to(p1.device), p1)
    return loss.mean()


