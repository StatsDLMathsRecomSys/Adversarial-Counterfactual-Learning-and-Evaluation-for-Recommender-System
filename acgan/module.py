"""Modules are to express the mathematical relationships between parameters.

Design note: The module shoudn't care about things like data transformations. It should be
as self-contained as possible. Dirty jobs should be done by the Model class which serves
as a bridge between reality(data) and the theory(module).
"""
from typing import List, Tuple, Any, Optional
from scipy import sparse as sp  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
from torch import nn  # type: ignore

class PopularModel(nn.Module):
    def __init__(self, pop_cnt:  np.ndarray, shrinkage: float = 0.5):
        super(PopularModel, self).__init__()
        pop_cnt_cp = pop_cnt.copy()
        pop_cnt_cp[pop_cnt_cp < 1] = 1
        rel_pop = (pop_cnt_cp / pop_cnt_cp.max()) ** shrinkage
        rel_pop = rel_pop.reshape(-1, 1)
        self.rep_pop_table = nn.Embedding(rel_pop.shape[0], 1)
        self.rep_pop_table.weight.data.copy_(torch.from_numpy(rel_pop))
        self.rep_pop_table.weight.requires_grad = False

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:  # type: ignore
        item_pop_score = self.rep_pop_table(item).squeeze(-1)
        return item_pop_score

    def get_device(self):
        return self.rep_pop_table.weight.device
        
class FactorModel(nn.Module):
    def __init__(self, user_num: int, item_num: int, factor_num: int) -> None:
        super(FactorModel, self).__init__()
        self.embed_user = nn.Embedding(user_num, factor_num, sparse=True)
        self.bias_user = nn.Embedding(user_num, 1, sparse=True)
        self.embed_item = nn.Embedding(item_num, factor_num, sparse=True)
        self.bias_item = nn.Embedding(item_num, 1, sparse=True)

        self.final_layer = nn.Linear(factor_num, 1, bias=True)
        #self.bias_global = nn.Parameter(torch.zeros(1))

        nn.init.kaiming_normal_(self.embed_user.weight)
        nn.init.kaiming_normal_(self.embed_item.weight)
        nn.init.zeros_(self.bias_item.weight)
        nn.init.zeros_(self.bias_user.weight)

    def affinity_vector(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:  # type: ignore
        vec_user = self.embed_user(user)
        vec_item = self.embed_item(item)
        prediction = (vec_user * vec_item)
        return prediction

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:  # type: ignore
        affinity_vec = self.affinity_vector(user, item)
        bias_user = self.bias_user(user).squeeze(-1)
        bias_item = self.bias_item(item).squeeze(-1)
        prediction = self.final_layer(affinity_vec).squeeze(-1)
        prediction += bias_item + bias_user
        return prediction

    def get_sparse_weight(self) -> List[torch.Tensor]:
        out = [self.embed_user.weight, self.bias_user.weight,
                self.embed_item.weight, self.bias_item.weight]
        return out

    def get_dense_weight(self) -> List[torch.Tensor]:
        out = []
        out.extend(self.final_layer.parameters())
        return out

    def get_l2(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        vec_user = self.embed_user(user)
        vec_item = self.embed_item(item)
        l2_loss = (vec_user ** 2).sum()
        l2_loss += (vec_item ** 2).sum()
        l2_loss += (self.final_layer.weight ** 2).sum()
        return l2_loss

    def get_device(self):
        return self.embed_item.weight.device


class BetaModel(nn.Module):
    def __init__(self, user_num: int, item_num: int) -> None:
        super(BetaModel, self).__init__()
        self.user_const = nn.Embedding(user_num, 1, sparse=True)
        self.item_const = nn.Embedding(item_num, 1, sparse=True)
        self.alpha = torch.nn.Parameter(torch.zeros(1))  # type: ignore
        self.beta = torch.nn.Parameter(torch.ones(1))  # type: ignore
        self.label_coef = torch.nn.Parameter(torch.zeros(1))  # type: ignore

        nn.init.zeros_(self.user_const.weight)
        nn.init.zeros_(self.item_const.weight)

    def forward(self, user: torch.Tensor, item: torch.Tensor, g_s: torch.Tensor, label: torch.Tensor) -> torch.Tensor:  # type: ignore
        #user_v = self.user_const(user).squeeze(-1)
        #item_v = self.item_const(item).squeeze(-1)
        #score = (self.alpha + self.beta * g_s + self.label_coef * label * g_s)
        score = (self.alpha + self.beta * g_s + self.label_coef * label * g_s)  # beta v2
        #score += user_v + item_v
        return score

    def get_sparse_weight(self) -> List[torch.Tensor]:
        out = [self.user_const.weight, self.item_const.weight]
        return out

    def get_dense_weight(self) -> List[torch.Tensor]:
        return [self.alpha, self.beta, self.label_coef]

    def get_l2(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        user_v = self.user_const(user).squeeze(-1)
        item_v = self.item_const(item).squeeze(-1)
        l2_loss = (user_v ** 2).sum()
        l2_loss += (item_v ** 2).sum()
        #l2_loss += (self.beta ** 2).sum()
        #l2_loss += (self.alpha ** 2).sum()
        #l2_loss += (self.label_coef ** 2).sum()
        return l2_loss


class MLPRecModel(nn.Module):
    def __init__(
        self,
        user_num: int,
        item_num: int,
        factor_num: int,
        layers_dim: List[int] = [
            32,
            16]):
        super(MLPRecModel, self).__init__()
        self.embed_user = nn.Embedding(user_num, factor_num, sparse=True)
        self.embed_item = nn.Embedding(item_num, factor_num, sparse=True)

        nn.init.kaiming_normal_(self.embed_user.weight)
        nn.init.kaiming_normal_(self.embed_item.weight)

        self.dense_layers = nn.ModuleList()
        assert(isinstance(layers_dim, list))
        input_dims = [2 * factor_num] + layers_dim
        for i in range(len(layers_dim)):
            self.dense_layers.append(
                nn.Linear(input_dims[i], layers_dim[i], bias=True))
        self.act_func = nn.ReLU()
        self.out_put_layer = nn.Linear(layers_dim[-1], 1, bias=True)

    def affinity_vector(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:  # type: ignore
        vec_user = self.embed_user(user)
        vec_item = self.embed_item(item)
        x = torch.cat([vec_user, vec_item], dim=-1)
        for linear_layer in self.dense_layers:
            x = linear_layer(x)
            x = self.act_func(x)
        return x

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.affinity_vector(user, item)
        prediction = self.out_put_layer(x).squeeze(-1)
        return prediction

    def get_device(self):
        return self.embed_item.weight.device

    def score(self, u_b: List[int], v_b: List[int]) -> np.ndarray:
        with torch.no_grad():
            device = self.embed_user.weight.device
            ubt = torch.LongTensor(u_b).to(device)
            vbt = torch.LongTensor(v_b).to(device)
            score = self.forward(ubt, vbt).cpu().numpy()
        return score

    def get_sparse_weight(self) -> List[torch.Tensor]:
        out = [self.embed_user.weight, self.embed_item.weight]
        return out

    def get_dense_weight(self) -> List[torch.Tensor]:
        out = []
        for layer in self.dense_layers:
            out.extend(layer.parameters())
        out.extend(self.out_put_layer.parameters())
        return out

    def get_l2(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        vec_user = self.embed_user(user)
        vec_item = self.embed_item(item)
        l2_loss = (vec_user ** 2).sum()
        l2_loss += (vec_item ** 2).sum()
        # for weight in self.get_dense_weight():
        #     l2_loss += (weight ** 2).sum()
        return l2_loss

class NCFModel(nn.Module):
    def __init__(self, user_num: int, item_num: int, factor_num: int, layers_dim: Optional[List[int]] = None):
        super(NCFModel, self).__init__()
        if layers_dim is None:
            layers_dim = [factor_num // 2, factor_num // 4]

        mlp_out_dim = layers_dim[-1]
        gmf_out_dim = factor_num - mlp_out_dim
        gmf_in_dim = gmf_out_dim
        self.mlp = MLPRecModel(user_num, item_num, factor_num // 2, layers_dim=layers_dim)
        self.gmf = FactorModel(user_num, item_num, gmf_in_dim)
        self.out_put_layer = nn.Linear(in_features=factor_num, out_features=1, bias=True)
        
    def affinity_vector(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        mlp_vec = self.mlp.affinity_vector(user, item)
        gmf_vec = self.gmf.affinity_vector(user, item)
        return torch.cat([mlp_vec, gmf_vec], dim=-1)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        x = self.affinity_vector(user, item)
        return self.out_put_layer(x).squeeze(-1)

    def get_sparse_weight(self):
        return self.mlp.get_sparse_weight() + self.gmf.get_sparse_weight()

    def get_dense_weight(self):
        return self.mlp.get_dense_weight() + self.gmf.get_dense_weight()

    def get_l2(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        l2 = self.mlp.get_l2(user, item)
        l2 += self.gmf.get_l2(user, item)
        l2 += (self.out_put_layer.weight ** 2).sum()
        return l2

    def get_device(self):
        return self.gmf.get_device()


class StructureNoise(nn.Module):
    def __init__(self, factor_num: int) -> None:
        super(StructureNoise, self).__init__()
        self.l1 = nn.Linear(2 * factor_num, factor_num)
        self.l2 = nn.Linear(factor_num, factor_num)
        self.l3 = nn.Linear(factor_num, 1)
        self.act = nn.ReLU()

    def forward(
            self,
            user_vec: torch.Tensor,
            item_vec: torch.Tensor) -> torch.Tensor:
        x = torch.cat([user_vec, item_vec], dim=-1)
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        x = self.act(self.l3(x)).squeeze(-1)
        return x


class NoiseFactor(nn.Module):
    def __init__(self, facotr_model: torch.nn.Module, factor_num: int) -> None:
        super(NoiseFactor, self).__init__()
        self.noise_model = StructureNoise(factor_num)
        self.facotr_model = facotr_model
        self.embed_item = self.facotr_model.embed_item

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:  # type: ignore
        prediction = self.facotr_model(user, item)
        with torch.no_grad():
            vec_user = self.facotr_model.embed_user(user)
            vec_item = self.facotr_model.embed_item(item)
            prediction += self.noise_model(vec_user, vec_item)
        return prediction

    def get_sparse_weight(self) -> List[torch.Tensor]:
        return []

    def get_dense_weight(self) -> List[torch.Tensor]:
        return []

    def get_l2(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        return self.facotr_model.get_l2(user, item)

    def get_device(self):
        return self.facotr_model.get_device()

class AttentionModel(nn.Module):
    def __init__(
            self,
            user_num: int,
            item_num: int,
            factor_num: int,
            max_len: int = 20, 
            num_heads: int = 2, 
            num_layer: int = 2) -> None:
        super(AttentionModel, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.factor_num = factor_num
        self.padding_idx = self.item_num
        self.max_len = max_len
        #self.embed_user = nn.Embedding(user_num, factor_num, sparse=True)
        self.embed_item = nn.Embedding(item_num + 1, factor_num, sparse=False, padding_idx=self.padding_idx)
        #self.target_item_embed = nn.Embedding(item_num + 1, factor_num, sparse=False, padding_idx=self.padding_idx)
        self.position_encode = nn.Embedding(max_len, factor_num, sparse=False)
        self.attention_list = nn.ModuleList()
        for _ in range(num_layer):
            self.attention_list.append(nn.MultiheadAttention(embed_dim=factor_num, num_heads=num_heads))
        self.output_affine = nn.Linear(factor_num, 1, bias=True)
    
    def get_device(self):
        return self.embed_item.weight.device

    def seq_vector(self, user_hist: torch.Tensor) -> torch.Tensor:
        """
        args:
            user: [B]
            item: [B]
            user_hist: [B, max_len]
        """
        hist_item_vec = self.embed_item(user_hist) # [B, max_len, factor_num]
        pos = torch.arange(self.max_len, device=self.get_device()).reshape(1, -1).repeat(hist_item_vec.shape[0], 1)
        # add positional encoding
        mask_item = (user_hist == self.padding_idx)
        attn_item_vec = hist_item_vec + self.position_encode(pos)
        attn_item_vec = attn_item_vec.transpose(1, 0)  #[max_len, B, factor_num]

        for atten_layer in self.attention_list:
            attn_item_vec, _ = atten_layer(
                query=attn_item_vec, 
                key=attn_item_vec, 
                value=attn_item_vec, 
                key_padding_mask=mask_item)
        # attn_item_vec - [max_len, B, factor_num]
        attn_item_vec = attn_item_vec.mean(dim=0) #[B, factor_num]
        return attn_item_vec
    
    def forward(self, items: torch.Tensor, user_hists: torch.Tensor) -> torch.Tensor:
        # items - [B, ord]
        assert(len(items.shape) == 2)
        assert(items.shape[0] == user_hists.shape[0])

        affinity_vec = self.seq_vector(user_hists) # [B, dim]
        affinity_vec = affinity_vec.unsqueeze(1).repeat(1, items.shape[1], 1) # [B, ord, dim]
        target_item_vec = self.embed_item(items) # - [B, ord, dim]
        #target_item_vec = self.target_item_embed(items) # - [B, ord, dim]
        score = self.output_affine(affinity_vec * target_item_vec) # [B, ord, 1]
        return score.squeeze(-1) # [B, ord]


    def get_dense_weight(self):
        return list(self.parameters())

    def get_sparse_weight(self):
        return []

    def get_l2(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        target_item_vec = self.embed_item(items)
        return (target_item_vec ** 2).sum() * 0
    
