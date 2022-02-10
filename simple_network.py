import torch
import torch.nn as nn


class SimpleNetwork(nn.Module):
    def __init__(self, params_in: int, layer_sizes: list(), act_funs: list(), calc_deriv: bool, device: torch.device):
        super(SimpleNetwork, self).__init__()
        
        self.in_out = layer_sizes
        self.in_out.insert(0, params_in)
        self.ins = self.in_out[0:-1]
        self.outs = self.in_out[1:]
        self.act_funs = act_funs
        self.layer_count = len(self.act_funs)
        self.calc_deriv = calc_deriv
        self.device = device
        
        assert len(self.ins) == len(self.act_funs)
        assert len(self.outs) == len(self.act_funs)
        assert all([f in [nn.Sigmoid, nn.Tanh] for f in self.act_funs])
        
        for layer_idx, (n_in, n_out, act_fun) in enumerate(zip(self.ins, self.outs, self.act_funs), 1):
            layer = nn.Linear(n_in, n_out, bias=True)
            self.add_module(f'fc{layer_idx}', layer)
            self.add_module(f'act{layer_idx}', self.act_funs[layer_idx-1]())

        self.requires_grad_(not calc_deriv)
        if calc_deriv:
            self.fwd_activation = {}
            def get_fwd_activation(name):
                def hook(model, input, output):
                    self.fwd_activation[name] = output.detach()
                return hook
            for name, module in self.named_modules():
                if name != '':
                    _ = module.register_forward_hook(get_fwd_activation(name))
        self.last_x = None
    
    def forward(self, x):
        x = x.view(-1, self.ins[0])
        if self.calc_deriv and self._recalculate_derivatives(x):
            self.layer_derivatives_1 = dict()
            self.layer_derivatives_2 = dict()
        for layer_idx in range(1, self.layer_count + 1):
            x = self._modules[f'fc{layer_idx}'](x)
            x = self._modules[f'act{layer_idx}'](x)
        return x
    
    def _recalculate_derivatives(self, x):
        if (self.last_x is not None) and (self.last_x.shape == x.shape) and (torch.all(torch.eq(self.last_x, x))):
            return False
        self.last_x = x
        return True
    
    def _get_analytic_derivative_1(self, actfun, x):
        if type(actfun) == nn.Sigmoid:
            return actfun(x) * (1 - actfun(x))
        elif type(actfun) == nn.Tanh:
            return 1 - actfun(x)**2
        else:
            raise NotImplementedError()

    def _get_analytic_derivative_2(self, actfun, x):
        if type(actfun) == nn.Sigmoid:
            return actfun(x) * (1 - actfun(x)) * (1 - 2*actfun(x))
        elif type(actfun) == nn.Tanh:
            return -2*actfun(x)*(1 - actfun(x)**2)
        else:
            raise NotImplementedError()
    
    def _calculate_layer_derivatives_1(self):
        for layer_idx in range(1, self.layer_count + 1):
            layer_weights = self._modules[f'fc{layer_idx}'].weight
            act_d1 = self._modules[f'act{layer_idx}']
            act_input = self.fwd_activation[f'fc{layer_idx}']
            layer_act = self._get_analytic_derivative_1(act_d1, act_input)
            self.layer_derivatives_1.update(
                {layer_idx: layer_act.view(layer_act.shape[0], layer_act.shape[1], 1) * layer_weights})
    
    def _calculate_subnetwork_jacobian(self, p, q, x):
        if (len(self.layer_derivatives_1) == 0) or self._recalculate_derivatives(x):
            self._calculate_layer_derivatives_1()
        if ((p == 1) and (q == 0)):
            return torch.eye(self.ins[0], device=self.device).repeat(self.layer_derivatives_1[1].shape[0], 1, 1)
        if ((p == self.layer_count + 1) and (q == self.layer_count)):
            return torch.eye(self.outs[-1], device=self.device).repeat(self.layer_derivatives_1[1].shape[0], 1, 1)
        else:
            if not ((p >= 1) and (p <= self.layer_count) and (q >= p) and (q <= self.layer_count)):
                raise ValueError()
            subjac = self.layer_derivatives_1[q]
            for l in range(self.layer_count - q + 1, self.layer_count - p + 1):
                subjac = torch.matmul(subjac, self.layer_derivatives_1[self.layer_count - l])
            return subjac
    
    def get_network_jacobian(self, x):
        x = x.view(-1, self.ins[0])
        _ = self.forward(x)
        return self._calculate_subnetwork_jacobian(1, self.layer_count, x)

    def get_network_hessian_slice(self, x, j):
        x = x.view(-1, self.ins[0])
        _ = self.forward(x)
        hessian_slice = torch.zeros(x.shape[0], self.outs[-1], self.ins[0])
        for l in range(1, self.layer_count + 1):
            phi_pre = self._calculate_subnetwork_jacobian(1, l-1, x)
            phi_post = self._calculate_subnetwork_jacobian(l+1, self.layer_count, x)
            layer_weights = self._modules[f'fc{l}'].weight
            act_d2 = self._modules[f'act{l}']
            act_input = self.fwd_activation[f'fc{l}']
            layer_act = self._get_analytic_derivative_2(act_d2, act_input)
            m = torch.matmul(layer_weights, phi_pre)
            phi = (layer_act.unsqueeze(2) * m[:, :, j].unsqueeze(2)) * layer_weights
            hessian_slice += torch.matmul(torch.matmul(phi_post, phi), phi_pre)
        return hessian_slice
    
    def get_network_hessians(self, x):
        x = x.view(-1, self.ins[0])
        _ = self.forward(x)
        hessians = torch.zeros(x.shape[0], self.ins[0], self.outs[-1], self.ins[0], device=self.device)
        for l in range(1, self.layer_count + 1):
            phi_pre = self._calculate_subnetwork_jacobian(1, l-1, x)
            phi_post = self._calculate_subnetwork_jacobian(l+1, self.layer_count, x)
            layer_weights = self._modules[f'fc{l}'].weight
            act_d2 = self._modules[f'act{l}']
            act_input = self.fwd_activation[f'fc{l}']
            layer_act = self._get_analytic_derivative_2(act_d2, act_input)
            m = torch.matmul(layer_weights, phi_pre)
            phi = (layer_act.unsqueeze(2) * m).transpose(1, 2).unsqueeze(3) * layer_weights
            hessians += torch.matmul(torch.matmul(phi_post.unsqueeze(1), phi), phi_pre.unsqueeze(1))
        return hessians
    
    def destroy_model(self):
        del self.layer_derivatives_1, self.layer_derivatives_2, self.fwd_activation, self._modules

