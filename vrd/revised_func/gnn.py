import torch
import torch.nn as nn
import scipy.io as scio
import os
class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim, n_node, n_edge_types):
        super(Propogator, self).__init__()

        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Tanh()
        )

    def forward(self, state_in, state_cur, A):

        a_in = torch.bmm(A, state_in)
        a = torch.cat((a_in, state_cur), 2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, r * state_cur), 2)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        return output


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self,cfg,state_dim=10,annotation_dim=1,n_edge_types=3,n_node=70,n_steps=5):
        super(GGNN, self).__init__()
        self.state_dim = state_dim
        self.annotation_dim = annotation_dim
        self.n_edge_types = n_edge_types
        self.n_node = n_node
        self.n_steps = n_steps

        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)

        self.in_fcs = AttrProxy(self, "in_")

        # Propogation Model
        self.propogator = Propogator(self.state_dim, self.n_node, self.n_edge_types)

        # Output Model
        self.out = nn.Sequential(
            nn.Linear(self.state_dim + self.annotation_dim, self.state_dim),
            nn.Tanh(),
            nn.Linear(self.state_dim, 1)
        )

        self.A_loc = scio.loadmat(os.getcwd()+cfg.CONFIG.location_anchors)
        self.A_loc  = torch.from_numpy(self.A_loc['A_sim']).float()
        self.A_loc = torch.div(self.A_loc,torch.sum(self.A_loc,1))

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self,annotation,devices):
        annotation = annotation.unsqueeze(-1)
        padding = torch.zeros(len(annotation), self.n_node, self.state_dim - self.annotation_dim).to(devices)
        prop_state = torch.cat((annotation, padding), 2)
        A=self.A_loc.unsqueeze(0).expand(len(annotation), self.n_node, self.n_node*self.n_edge_types).to(devices)
        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state))

            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)
            prop_state = self.propogator(in_states, prop_state, A)

        join_state = torch.cat((prop_state, annotation), 2)
        output = self.out(join_state)
        output = output.sum(2)

        return output
