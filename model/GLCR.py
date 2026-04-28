import torch
import torch.nn as nn

from layer.GCN import GCN


class GLCRModel(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, dropout):
        super(GLCRModel, self).__init__()
        self.encoder = GCN(in_channels, hidden_dim, out_channels, dropout)

    def forward(self, data):
        return self.encoder(data)


class GLCRClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        dropout=0.1,
        subgraph_dim=0,
        latent_mediator_count=16,
        mediator_dropout=0.2,
        utility_gate_floor=0.50,
    ):
        super(GLCRClassifier, self).__init__()
        if subgraph_dim <= 0:
            raise ValueError("GLCR mainline requires subgraph context features.")

        self.mediator_dropout = mediator_dropout
        self.utility_gate_floor = float(utility_gate_floor)
        self.projected_subgraph_dim = max(32, input_dim // 2)

        pair_dim = input_dim * 4
        utility_input_dim = input_dim * 2 + 4 + 2 + num_classes

        self.subgraph_proj = nn.Sequential(
            nn.LayerNorm(subgraph_dim),
            nn.Linear(subgraph_dim, self.projected_subgraph_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mlp = nn.Sequential(
            nn.Linear(pair_dim + self.projected_subgraph_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes),
        )

        self.direct_encoder = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.direct_output = nn.Linear(input_dim, num_classes)
        self.bridge_output = nn.Linear(input_dim, num_classes)

        self.latent_mediator_bank = nn.Parameter(torch.randn(latent_mediator_count, input_dim) * 0.02)
        self.latent_query = nn.Linear(pair_dim, input_dim)
        self.latent_gate = nn.Sequential(
            nn.Linear(input_dim * 2 + 4, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )
        self.latent_norm = nn.LayerNorm(input_dim)
        self.latent_context_proj = nn.Sequential(
            nn.LayerNorm(self.projected_subgraph_dim),
            nn.Linear(self.projected_subgraph_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        bridge_hidden_dim = max(32, input_dim // 2)
        self.pair_encoder = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.bridge_node_proj = nn.Sequential(
            nn.Linear(input_dim, bridge_hidden_dim),
            nn.ReLU(),
        )
        self.bridge_query = nn.Linear(pair_dim, bridge_hidden_dim)
        self.bridge_feat_proj = nn.Sequential(
            nn.Linear(3, bridge_hidden_dim),
            nn.ReLU(),
        )
        self.bridge_prior_scale = nn.Parameter(torch.tensor(1.0))
        self.mediator_token_proj = nn.Sequential(
            nn.LayerNorm(bridge_hidden_dim * 3 + 1),
            nn.Linear(bridge_hidden_dim * 3 + 1, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mediator_token_score = nn.Linear(input_dim, 1)
        self.mediator_token_value = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.bridge_refine_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),
        )
        self.bridge_refine_gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )
        self.bridge_stats_proj = nn.Sequential(
            nn.Linear(4, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.bridge_stats_gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )
        self.mediator_confidence = nn.Sequential(
            nn.Linear(input_dim + 4, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid(),
        )

        local_input_dim = self.projected_subgraph_dim + input_dim
        self.local_subgraph_encoder = nn.Sequential(
            nn.LayerNorm(local_input_dim),
            nn.Linear(local_input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.global_to_local_gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )
        self.local_to_global_gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )
        self.bridge_message_proj = nn.Sequential(
            nn.LayerNorm(input_dim * 2),
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.bridge_confidence = nn.Sequential(
            nn.Linear(input_dim * 2 + 4 + 1, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
            nn.Sigmoid(),
        )
        self.pair_to_local_delta = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),
        )
        self.local_to_pair_delta = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),
        )
        self.pair_to_local_delta_gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )
        self.local_to_pair_delta_gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )
        self.context_token_projector = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.context_token_scorer = nn.Linear(input_dim, 1)
        self.utility_gate = nn.Sequential(
            nn.Linear(utility_input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        nn.init.constant_(self.bridge_confidence[-2].bias, -2.0)
        nn.init.constant_(self.global_to_local_gate[-2].bias, 1.0)
        nn.init.constant_(self.local_to_global_gate[-2].bias, -1.0)

    def forward(
        self,
        x_i,
        x_j,
        subgraph_emb=None,
        bridge_nodes=None,
        bridge_mask=None,
        bridge_prior=None,
        bridge_stats=None,
        bridge_feat=None,
        return_explain=False,
        return_aux=False,
        perturb_explicit=False,
    ):
        pair_features = torch.cat([x_i, x_j, torch.abs(x_i - x_j), x_i * x_j], dim=1)
        projected_subgraph = self._project_subgraph(subgraph_emb, pair_features)
        stats = self._bridge_stats(bridge_stats, pair_features)

        base_logits = self.mlp(torch.cat([pair_features, projected_subgraph], dim=1))
        pair_state = self.pair_encoder(pair_features)
        explicit_state, explicit_gate, explicit_confidence = self._build_explicit_state(
            pair_features,
            projected_subgraph,
            bridge_nodes,
            bridge_mask,
            bridge_prior,
            stats,
            bridge_feat,
        )

        direct_repr = self.direct_encoder(pair_features)
        base_local_seed = self.local_subgraph_encoder(torch.cat([projected_subgraph, explicit_state], dim=1))
        base_latent_residual = self._build_latent_state(
            pair_features,
            pair_state,
            projected_subgraph,
            base_local_seed,
            stats,
            explicit_confidence,
        )
        base_bridge_repr = self.bridge_message_proj(torch.cat([base_local_seed, explicit_state], dim=1))
        base_bridge_repr = base_bridge_repr + (1.0 - explicit_confidence) * base_latent_residual
        base_bridge_logits = self.bridge_output(base_bridge_repr)

        local_seed = self._build_micro_subgraph_state(explicit_state, base_local_seed)

        global_to_local = self.global_to_local_gate(torch.cat([pair_state, local_seed], dim=1))
        local_comm = global_to_local * pair_state + (1.0 - global_to_local) * local_seed

        pair_delta = self.pair_to_local_delta(torch.cat([pair_state, explicit_state], dim=1))
        pair_delta_gate = self.pair_to_local_delta_gate(torch.cat([pair_state, explicit_state], dim=1))
        staged_local = local_seed + pair_delta_gate * pair_delta
        bridge_delta = self.local_to_pair_delta(torch.cat([pair_state, staged_local], dim=1))
        bridge_delta_gate = self.local_to_pair_delta_gate(torch.cat([pair_state, staged_local], dim=1))
        local_comm = 0.5 * local_comm + 0.5 * staged_local
        global_comm = pair_state + bridge_delta_gate * bridge_delta

        latent_residual = self._build_latent_state(
            pair_features,
            global_comm,
            projected_subgraph,
            local_comm,
            stats,
            explicit_confidence,
        )
        bridge_repr = self.bridge_message_proj(torch.cat([local_comm, explicit_state], dim=1))
        bridge_repr = bridge_repr + (1.0 - explicit_confidence) * latent_residual
        bridge_confidence = self.bridge_confidence(
            torch.cat([global_comm, bridge_repr, stats, explicit_confidence], dim=1)
        )
        bridge_logits = self.bridge_output(bridge_repr)
        global_logits = self.direct_output(global_comm)

        utility_gap = torch.abs(global_logits - bridge_logits)
        utility_input = torch.cat(
            [global_comm, bridge_repr, stats, explicit_confidence, bridge_confidence, utility_gap],
            dim=1,
        )
        utility = self.utility_gate(utility_input)
        utility_gain = self.utility_gate_floor + (1.0 - self.utility_gate_floor) * utility

        bridge_term = (bridge_confidence * utility_gain) * base_bridge_logits
        local_residual_logits = utility_gain * (bridge_logits - base_bridge_logits)
        bridge_contribution = bridge_term + local_residual_logits
        final_logits = global_logits + bridge_contribution + self.residual_scale * base_logits

        perturbed_logits = None
        if perturb_explicit and bridge_mask is not None and bridge_mask.numel() > 0:
            perturbed_logits = self._compute_perturbed_logits(
                pair_features=pair_features,
                projected_subgraph=projected_subgraph,
                stats=stats,
                pair_state=pair_state,
                explicit_confidence=explicit_confidence,
                global_to_local=global_to_local,
                global_comm=global_comm,
                latent_residual=latent_residual,
                base_logits=base_logits,
                base_bridge_logits=base_bridge_logits,
                bridge_nodes=bridge_nodes,
                bridge_mask=bridge_mask,
                bridge_prior=bridge_prior,
                bridge_feat=bridge_feat,
            )

        if return_explain:
            return final_logits, {
                "base_logits": global_logits.detach(),
                "final_logits": final_logits.detach(),
                "residual_logits": bridge_contribution.detach(),
                "bridge_attn": explicit_gate.detach() if explicit_gate is not None else None,
                "bridge_prior": bridge_prior.detach() if bridge_prior is not None else None,
                "gate": bridge_confidence.detach(),
                "utility": utility.detach(),
                "utility_gain": utility_gain.detach(),
            }
        if return_aux:
            return final_logits, {
                "direct_repr": direct_repr,
                "explicit_repr": local_seed,
                "pre_comm_local_seed": base_local_seed,
                "latent_repr": latent_residual,
                "perturbed_logits": perturbed_logits,
                "explicit_gate": explicit_gate,
                "explicit_confidence": explicit_confidence,
                "latent_confidence": 1.0 - explicit_confidence,
                "bridge_repr": bridge_repr,
                "bridge_confidence": bridge_confidence,
                "utility": utility,
                "utility_gain": utility_gain,
                "base_bridge_logits": base_bridge_logits,
                "global_comm": global_comm,
                "local_comm": local_comm,
            }
        return final_logits

    def _compute_perturbed_logits(
        self,
        pair_features,
        projected_subgraph,
        stats,
        pair_state,
        explicit_confidence,
        global_to_local,
        global_comm,
        latent_residual,
        base_logits,
        base_bridge_logits,
        bridge_nodes,
        bridge_mask,
        bridge_prior,
        bridge_feat,
    ):
        dropped_explicit, _, dropped_confidence = self._build_explicit_state(
            pair_features,
            projected_subgraph,
            bridge_nodes,
            bridge_mask,
            bridge_prior,
            stats,
            bridge_feat,
            perturb=True,
        )
        dropped_base_local_seed = self.local_subgraph_encoder(torch.cat([projected_subgraph, dropped_explicit], dim=1))
        dropped_base_latent = self._build_latent_state(
            pair_features,
            pair_state,
            projected_subgraph,
            dropped_base_local_seed,
            stats,
            dropped_confidence,
        )
        dropped_base_bridge_repr = self.bridge_message_proj(
            torch.cat([dropped_base_local_seed, dropped_explicit], dim=1)
        )
        dropped_base_bridge_repr = dropped_base_bridge_repr + (1.0 - dropped_confidence) * dropped_base_latent
        dropped_base_bridge_logits = self.bridge_output(dropped_base_bridge_repr)

        dropped_local_seed = self._build_micro_subgraph_state(dropped_explicit, dropped_base_local_seed)
        dropped_local_comm = global_to_local * global_comm + (1.0 - global_to_local) * dropped_local_seed
        dropped_pair_delta = self.pair_to_local_delta(torch.cat([pair_state, dropped_explicit], dim=1))
        dropped_pair_delta_gate = self.pair_to_local_delta_gate(torch.cat([pair_state, dropped_explicit], dim=1))
        dropped_staged_local = dropped_local_seed + dropped_pair_delta_gate * dropped_pair_delta
        dropped_local_comm = 0.5 * dropped_local_comm + 0.5 * dropped_staged_local

        dropped_bridge_repr = self.bridge_message_proj(torch.cat([dropped_local_comm, dropped_explicit], dim=1))
        dropped_bridge_repr = dropped_bridge_repr + (1.0 - dropped_confidence) * latent_residual
        dropped_bridge_confidence = self.bridge_confidence(
            torch.cat([global_comm, dropped_bridge_repr, stats, dropped_confidence], dim=1)
        )
        dropped_bridge_logits = self.bridge_output(dropped_bridge_repr)

        dropped_gap = torch.abs(global_comm.new_zeros(()) + self.direct_output(global_comm) - dropped_bridge_logits)
        dropped_utility_input = torch.cat(
            [global_comm, dropped_bridge_repr, stats, dropped_confidence, dropped_bridge_confidence, dropped_gap],
            dim=1,
        )
        dropped_utility = self.utility_gate(dropped_utility_input)
        dropped_utility_gain = self.utility_gate_floor + (1.0 - self.utility_gate_floor) * dropped_utility
        return (
            self.direct_output(global_comm)
            + (dropped_bridge_confidence * dropped_utility_gain) * dropped_base_bridge_logits
            + dropped_utility_gain * (dropped_bridge_logits - dropped_base_bridge_logits)
            + self.residual_scale * base_logits
        )

    def _project_subgraph(self, subgraph_emb, ref_tensor):
        if subgraph_emb is None:
            return ref_tensor.new_zeros(ref_tensor.size(0), self.projected_subgraph_dim)
        return self.subgraph_proj(subgraph_emb)

    def _bridge_stats(self, bridge_stats, ref_tensor):
        if bridge_stats is None:
            return ref_tensor.new_zeros(ref_tensor.size(0), 4)
        return bridge_stats

    def _build_latent_state(self, pair_features, pair_state, projected_subgraph, local_comm, bridge_stats, explicit_confidence):
        latent_query = self.latent_query(pair_features)
        latent_scores = torch.matmul(latent_query, self.latent_mediator_bank.t()) / (latent_query.size(1) ** 0.5)
        latent_weights = torch.softmax(latent_scores, dim=1)
        latent_summary = torch.matmul(latent_weights, self.latent_mediator_bank)
        context_state = pair_state + self.latent_context_proj(projected_subgraph)
        latent_gate = self.latent_gate(torch.cat([context_state, latent_summary, bridge_stats], dim=1))
        latent_state = latent_gate * (context_state - explicit_confidence * local_comm) + (1.0 - latent_gate) * latent_summary
        return self.latent_norm(latent_state)

    def _build_micro_subgraph_state(self, explicit_state, local_seed):
        tokens = [
            self.context_token_projector(explicit_state),
            self.context_token_projector(local_seed),
        ]
        token_stack = torch.stack(tokens, dim=1)
        token_scores = self.context_token_scorer(token_stack).squeeze(-1)
        token_weights = torch.softmax(token_scores, dim=1)
        pooled = (token_stack * token_weights.unsqueeze(-1)).sum(dim=1)
        return 0.5 * local_seed + 0.5 * pooled

    def _build_explicit_state(
        self,
        pair_features,
        projected_subgraph,
        bridge_nodes,
        bridge_mask,
        bridge_prior,
        bridge_stats,
        bridge_feat,
        perturb=False,
    ):
        if bridge_nodes is None or bridge_mask is None:
            zero_state = pair_features.new_zeros(pair_features.size(0), self.direct_output.in_features)
            zero_conf = pair_features.new_zeros(pair_features.size(0), 1)
            return zero_state, None, zero_conf

        active_mask = bridge_mask
        if perturb:
            active_mask = (torch.rand_like(bridge_mask) > self.mediator_dropout).float() * bridge_mask
            fallback = (active_mask.sum(dim=1, keepdim=True) == 0).float()
            active_mask = torch.maximum(active_mask, fallback * bridge_mask[:, :1].repeat(1, bridge_mask.size(1)))

        bridge_key = self.bridge_node_proj(bridge_nodes)
        bridge_feat_state = self.bridge_feat_proj(bridge_feat) if bridge_feat is not None else torch.zeros_like(bridge_key)
        bridge_query = self.bridge_query(pair_features).unsqueeze(1).expand_as(bridge_key)
        prior_token = bridge_prior.unsqueeze(-1) if bridge_prior is not None else bridge_key.new_zeros(
            bridge_key.size(0),
            bridge_key.size(1),
            1,
        )

        token_input = torch.cat([bridge_key, bridge_feat_state, bridge_query, prior_token], dim=-1)
        token_state = self.mediator_token_proj(token_input)

        query_token = self.pair_encoder(pair_features).unsqueeze(1).expand_as(token_state)
        refine_gate = self.bridge_refine_gate(torch.cat([token_state, query_token], dim=-1))
        refine_delta = self.bridge_refine_mlp(torch.cat([token_state, query_token], dim=-1))
        token_state = token_state + refine_gate * refine_delta

        learned_scores = self.mediator_token_score(token_state).squeeze(-1)
        prior_scores = 0.0
        if bridge_prior is not None:
            prior_scores = 0.5 * self.bridge_prior_scale * torch.log(bridge_prior.clamp_min(1e-8))
        attn_logits = (learned_scores + prior_scores).masked_fill(active_mask <= 0, -1e9)
        attn = torch.softmax(attn_logits, dim=1)
        attn = attn * active_mask
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-8)

        token_value = self.mediator_token_value(token_state)
        explicit_state = (token_value * attn.unsqueeze(-1)).sum(dim=1)

        stats_residual = self.bridge_stats_proj(bridge_stats)
        pair_anchor = self.pair_encoder(pair_features)
        stats_gate = self.bridge_stats_gate(torch.cat([pair_anchor, explicit_state], dim=1))
        explicit_state = explicit_state + 0.25 * stats_gate * stats_residual
        explicit_state = explicit_state + self.latent_context_proj(projected_subgraph)
        explicit_state = self.latent_norm(explicit_state)
        confidence = self.mediator_confidence(torch.cat([explicit_state, bridge_stats], dim=1))
        return explicit_state, attn, confidence
