import torch


def vendi_score(embeddings: torch.Tensor, eps: float = 1e-12):

    embeddings = embeddings / embeddings.norm(p=2, dim=1, keepdim=True)
    n = embeddings.size(0)

    sim_matrix = embeddings @ embeddings.T / n 

    eigvals = torch.linalg.eigvalsh(sim_matrix + eps * torch.eye(n, device=embeddings.device))
    eigvals = torch.clamp(eigvals, min=eps)

    entropy = torch.exp(- torch.sum(eigvals * torch.log(eigvals)))
    return entropy.item()



def avg_cosine(embeddings: torch.Tensor):

    embeddings = embeddings / embeddings.norm(p=2, dim=1, keepdim=True)
    n = embeddings.size(0)

    sim_matrix = embeddings @ embeddings.T
    sim_matrix.fill_diagonal_(0)

    avg_score = sim_matrix.sum() / (n * (n-1))
    return (1 - avg_score).item()

